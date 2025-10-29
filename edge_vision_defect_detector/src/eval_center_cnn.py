import os, yaml, numpy as np, cv2, torch
from tqdm import tqdm
from pycocotools.coco import COCO
from .utils import nms, xywh_to_xyxy, clip_box, extract_patch
from .candidate import detect_candidates
from .models_cnn import build_model

def centers(boxes): 
    return np.array([((x1+x2)/2.0,(y1+y2)/2.0) for x1,y1,x2,y2 in boxes], dtype=np.float32) if boxes else np.zeros((0,2),np.float32)

def score_image(img, model, device, cfg):
    cand = cfg["candidate"]
    boxes, _ = detect_candidates(
        img, cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
        cand.get("mask_top_ratio"), cand.get("min_contrast", 0.0))
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
    keep = [i for i,s in enumerate(scores) if s >= cfg.get("score_thresh", 0.0)]
    boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
    keep = nms(boxes, scores, cfg["nms_iou"])
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def main(cfg_path, arch):
    cfg = yaml.safe_load(open(cfg_path))
    dev=torch.device("cpu")
    ckpt = torch.load(os.path.join(cfg["model_dir"], f"cnn_{arch}.pt"), map_location=dev)
    model = build_model(ckpt["arch"]).to(dev); model.load_state_dict(ckpt["state"]); model.eval()

    coco = COCO(cfg["val_coco"])
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    tol = float(cfg.get("center_tol", 6))

    TP=FP=0; n_gt=0
    for img_id in tqdm(coco.getImgIds(), desc="Eval(center,CNN)"):
        info = coco.loadImgs([img_id])[0]
        p = os.path.join(os.path.dirname(cfg["val_coco"]), info["file_name"])
        img = cv2.imread(p); 
        if img is None: continue
        H,W = img.shape[:2]
        gt=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        n_gt += len(gt)
        det_b, det_s = score_image(img, model, dev, cfg)
        if not gt:
            FP += len(det_b); continue
        gt_c = centers(gt); det_c = centers(det_b)
        used=set()
        for cx,cy in det_c:
            if gt_c.size==0: FP+=1; continue
            d2 = np.sum((gt_c - np.array([cx,cy]))**2, axis=1)
            j = int(np.argmin(d2))
            if np.sqrt(d2[j]) <= tol and j not in used:
                TP += 1; used.add(j)
            else:
                FP += 1
    prec = TP / max(1, TP+FP); rec = TP / max(1, n_gt)
    print(f"Center@{tol:.1f}px  P={prec:.3f} R={rec:.3f}  (TP={TP}, FP={FP}, GT={n_gt})")

if __name__ == "__main__":
    import sys
    arch="large"
    args=sys.argv[1:] or ["configs/cnn.yaml"]
    if len(args)>=2 and args[0] in ["--arch","-a"]:
        arch=args[1]; cfg=args[2] if len(args)>=3 else "configs/cnn.yaml"
    else:
        cfg=args[0]
    main(cfg, arch)
