# src/validate_rfdetr.py
import os, yaml, cv2, numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pathlib import Path
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

VARIANTS = {
    "nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase,
    "medium": RFDETRMedium, "large": RFDETRLarge,
}

def load_model(cfg):
    d = cfg.get("rfdetr", {})
    variant = d.get("variant", "small").lower()
    Model = VARIANTS[variant]
    out_dir = os.path.join(cfg.get("model_dir","data/models"), d.get("output_subdir", f"rfdetr_{variant}"))
    best = os.path.join(out_dir, "checkpoint_best_total.pth")
    if os.path.isfile(best):
        return Model(pretrain_weights=best)
    return Model()

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(0, ax2-ax1)*max(0, ay2-ay1)
    area_b = max(0, bx2-bx1)*max(0, by2-by1)
    union = area_a + area_b - inter + 1e-9
    return inter/union

def predict(model, img, conf, iou, imgsz):
    det = model.predict(img, confidence=conf, iou_threshold=iou, imgsz=imgsz)
    if isinstance(det, list) and len(det) > 0 and hasattr(det[0], "xyxy"):
        det = det[0]
    if hasattr(det, "xyxy") and hasattr(det, "confidence"):
        return det.xyxy.astype(float), det.confidence.astype(float)
    return np.zeros((0,4), float), np.zeros((0,), float)

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model = load_model(cfg)
    conf0 = float(cfg.get("infer", {}).get("confidence", 0.5))
    iou_nms = float(cfg.get("infer", {}).get("iou", 0.5))
    imgsz = int(cfg.get("infer", {}).get("imgsz", 896))
    eval_iou = float(cfg.get("eval_iou", 0.20))

    coco = COCO(os.path.join(cfg["dataset_dir"], "valid", "_annotations.coco.json"))
    img_ids = coco.getImgIds()

    # precompute predictions
    preds = []
    gts = []
    print("Precomputing predictions...")
    for img_id in tqdm(img_ids):
        info = coco.loadImgs([img_id])[0]
        p = os.path.join(os.path.dirname(coco.dataset["annotations"][0]["file_name"]) if False else os.path.join(cfg["dataset_dir"], "valid"), info["file_name"])
        p = os.path.join(os.path.join(cfg["dataset_dir"], "valid"), info["file_name"])
        img = cv2.imread(p)
        if img is None:
            preds.append((np.zeros((0,4)), np.zeros((0,))))
            gts.append([])
            continue
        boxes, scores = predict(model, img, conf0, iou_nms, imgsz)
        preds.append((boxes, scores))

        H,W = img.shape[:2]
        gt_boxes = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get("iscrowd", 0): continue
            x,y,w,h = ann["bbox"]
            gt_boxes.append([x, y, x+w, y+h])
        gts.append(gt_boxes)

    # sweep thresholds and report P/R/F1
    best = (0.0, None)
    for th in np.linspace(0.05, 0.95, 19):
        TP=FP=0; n_gt=sum(len(x) for x in gts)
        for (boxes, scores), gt in zip(preds, gts):
            keep = scores >= th
            b = boxes[keep]
            matched=set()
            for bb in b.tolist():
                if not gt:
                    FP+=1; continue
                ious = [iou_xyxy(bb, g) for g in gt]
                m = int(np.argmax(ious)) if ious else -1
                if ious and ious[m] >= eval_iou and m not in matched:
                    TP+=1; matched.add(m)
                else:
                    FP+=1
        prec = TP / max(1, TP+FP); rec = TP / max(1, n_gt)
        f1 = (2*prec*rec)/(prec+rec+1e-9)
        print(f"th={th:.2f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        if f1 > best[0]: best = (f1, th)

    print(f"\nBest F1={best[0]:.3f} at score_thresh={best[1]:.2f}")

if __name__ == "__main__":
    import argparse, numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", nargs="?", default="configs/rfdetr.yaml")
    args = ap.parse_args()
    main(args.cfg)
