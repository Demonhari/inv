import os, yaml, numpy as np, cv2, torch, glob
from tqdm import tqdm
from pycocotools.coco import COCO
from .candidate import detect_candidates
from .utils import xywh_to_xyxy, clip_box, extract_patch, box_iou, ensure_dir
from .models_cnn import build_model

def main(cfg_path, arch="large", iou_thresh=0.05, score_thresh=0.0, max_per_img=200):
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cpu")
    ckpt = torch.load(os.path.join(cfg["model_dir"], f"cnn_{arch}.pt"), map_location=dev)
    model = build_model(ckpt["arch"]).to(dev); model.load_state_dict(ckpt["state"]); model.eval()

    coco = COCO(cfg["train_coco"])
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    out_dir = os.path.join(cfg["patch_root"], "train_neg_hard")
    ensure_dir(out_dir)
    cand = cfg["candidate"]

    saved = 0
    for img_id in tqdm(coco.getImgIds(), desc="Mine hard negatives"):
        info = coco.loadImgs([img_id])[0]
        p = os.path.join(os.path.dirname(cfg["train_coco"]), info["file_name"])
        img = cv2.imread(p)
        if img is None: continue
        H,W = img.shape[:2]
        gt=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))

        boxes,_ = detect_candidates(
            img, cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
            cand.get("mask_top_ratio"), cand.get("min_contrast",0.0))
        # batch-score
        feats=[]; idx=[]
        for i,b in enumerate(boxes):
            pch = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
            if pch is None: continue
            g = cv2.cvtColor(pch, cv2.COLOR_BGR2GRAY)
            feats.append(torch.from_numpy(g).float().div(255.0).unsqueeze(0).unsqueeze(0))
            idx.append(i)
        if not feats: continue
        x = torch.cat(feats,0).to(dev)
        with torch.no_grad(): logits = model(x).cpu().numpy().tolist()
        # select top-scoring negatives
        order = np.argsort(-np.array(logits))[:max_per_img]
        kept = 0
        for k in order:
            b = boxes[idx[k]]; s = float(logits[k])
            if s < score_thresh: break
            # reject if overlaps any GT
            if any(box_iou(b, g) >= iou_thresh for g in gt): 
                continue
            pch = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
            if pch is None: continue
            outp = os.path.join(out_dir, f"{info['file_name'].replace('/','_')}_{idx[k]:05d}.png")
            cv2.imwrite(outp, pch)
            saved += 1; kept += 1
            if kept >= max_per_img: break
    print(f"Saved {saved} hard negatives to {out_dir}")

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv)>1 else "configs/cnn.yaml"
    arch = sys.argv[2] if len(sys.argv)>2 else "large"
    main(cfg, arch)
