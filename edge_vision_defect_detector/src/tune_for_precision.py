import os, yaml, numpy as np, cv2, torch
from tqdm import tqdm
from pycocotools.coco import COCO
from .utils import nms, xywh_to_xyxy, clip_box, box_iou, extract_patch
from .candidate import detect_candidates
from .models_cnn import build_model

def score_image(img, model, device, cfg):
    cand = cfg["candidate"]
    boxes,_ = detect_candidates(
        img, cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
        cand.get("mask_top_ratio"), cand.get("min_contrast", 0.0)
    )
    feats=[]; idx=[]
    for i,b in enumerate(boxes):
        p = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
        if p is None: continue
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        feats.append(torch.from_numpy(g).float().div(255.0).unsqueeze(0).unsqueeze(0))
        idx.append(i)
    if not feats: return [], []
    x = torch.cat(feats,0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().tolist()
    boxes = [boxes[i] for i in idx]; scores = logits
    return boxes, scores

def main(cfg_path, arch, target_prec=0.90):
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cpu")
    ckpt = torch.load(os.path.join(cfg["model_dir"], f"cnn_{arch}.pt"), map_location=dev)
    model = build_model(ckpt["arch"]).to(dev); model.load_state_dict(ckpt["state"]); model.eval()

    coco = COCO(cfg["val_coco"])
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    eval_iou = float(cfg["eval_iou"]); nms_iou = float(cfg["nms_iou"])
    max_k = int(cfg.get("max_dets_per_image", 100000))

    img_ids = coco.getImgIds()
    # precompute per-image detections once
    all_boxes=[]; all_scores=[]; per_img_gt=[]
    for img_id in tqdm(img_ids, desc="Precompute"):
        info = coco.loadImgs([img_id])[0]
        p = os.path.join(os.path.dirname(cfg["val_coco"]), info["file_name"])
        img = cv2.imread(p)
        if img is None:
            all_boxes.append([]); all_scores.append([]); per_img_gt.append([]); continue
        H,W = img.shape[:2]
        gt=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        per_img_gt.append(gt)
        b,s = score_image(img, model, dev, cfg)
        all_boxes.append(b); all_scores.append(s)

    best = {"th": None, "prec": 0.0, "rec": 0.0}
    # sweep logits thresholds; widen/narrow if needed
    for th in np.linspace(-0.5, 1.0, 31):
        TP=FP=0; n_gt=sum(len(g) for g in per_img_gt)
        for boxes, scores, gt in zip(all_boxes, all_scores, per_img_gt):
            keep = [i for i,v in enumerate(scores) if v >= th]
            b = [boxes[i] for i in keep]; s = [scores[i] for i in keep]
            if b:
                k = nms(b, s, nms_iou)
                b = [b[i] for i in k]; s = [s[i] for i in k]
                if len(s) > max_k:
                    order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:max_k]
                    b = [b[i] for i in order]; s = [s[i] for i in order]
            matched=set()
            for bb, ss in sorted(zip(b,s), key=lambda x: -x[1]):
                if not gt: FP+=1; continue
                ious = [box_iou(bb, g) for g in gt]
                if not ious: FP+=1; continue
                m = int(np.argmax(ious))
                if ious[m] >= eval_iou and m not in matched:
                    TP += 1; matched.add(m)
                else:
                    FP += 1
        prec = TP / max(1, TP+FP); rec = TP / max(1, n_gt)
        print(f"th={th:.2f}  P={prec:.3f} R={rec:.3f}")
        # keep best recall among thresholds that meet precision target
        if prec >= target_prec and rec >= best["rec"]:
            best = {"th": float(th), "prec": float(prec), "rec": float(rec)}

    if best["th"] is None:
        print("\nNo threshold reached the target precision. Consider tightening proposals (Step 2).")
    else:
        print(f"\nChosen threshold for precision >= {target_prec:.2f}: {best['th']:.2f}")
        print(f"Estimated Val: P={best['prec']:.3f}, R={best['rec']:.3f}")
        # write back to config
        cfg["score_thresh"] = best["th"]
        open(cfg_path,'w').write(yaml.dump(cfg, sort_keys=False))
        print(f"Updated score_thresh in {cfg_path}")
if __name__ == "__main__":
    import sys
    # usage:
    #   python -m src.tune_for_precision configs/cnn.yaml       (defaults to 'large')
    #   python -m src.tune_for_precision configs/cnn.yaml xl
    args = sys.argv[1:] or ["configs/cnn.yaml"]
    if len(args) >= 2:
        cfg, arch = args[0], args[1]
    else:
        cfg, arch = args[0], "large"
    main(cfg, arch, target_prec=0.90)
