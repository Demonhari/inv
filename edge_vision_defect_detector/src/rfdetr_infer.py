# src/rfdetr_infer.py
import os, glob, yaml, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
import supervision as sv

VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}

def load_model(cfg):
    d = cfg.get("rfdetr", {})
    variant = d.get("variant", "small").lower()
    Model = VARIANTS[variant]
    # If you trained already, point pretrain_weights to best checkpoint
    out_dir = os.path.join(cfg.get("model_dir","data/models"), d.get("output_subdir", f"rfdetr_{variant}"))
    best = os.path.join(out_dir, "checkpoint_best_total.pth")
    pre = best if os.path.isfile(best) else None
    if pre:
        print(f"Loading fine-tuned weights: {best}")
        return Model(pretrain_weights=best)
    else:
        print("Loading COCO pretrain (no fine-tuned weights found).")
        return Model()

def predict_image(model, img_bgr, conf=0.5, iou=0.5, imgsz=896):
    # RF-DETR works with BGR ndarray too; returns supervision.Detections
    det = model.predict(
        img_bgr,
        confidence=conf,
        iou_threshold=iou,
        imgsz=imgsz
    )
    # Normalize to (xyxy: Nx4 float, scores: N, class_id: N)
    if isinstance(det, list) and len(det) > 0 and hasattr(det[0], "xyxy"):
        det = det[0]
    if hasattr(det, "xyxy") and hasattr(det, "confidence"):
        boxes = det.xyxy.astype(float)
        scores = det.confidence.astype(float)
        cls_id = getattr(det, "class_id", None)
    else:
        raise RuntimeError("Unrecognized RF-DETR output format")

    return boxes, scores, cls_id

def draw_and_save(img_bgr, boxes, scores, names, out_path, thresh=0.5):
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        if s < thresh: continue
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f"{names[0]} {s:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img_bgr)

def infer_dir(cfg_path, in_dir, out_dir):
    cfg = yaml.safe_load(open(cfg_path))
    model = load_model(cfg)
    os.makedirs(out_dir, exist_ok=True)

    conf = float(cfg.get("infer", {}).get("confidence", 0.5))
    iou  = float(cfg.get("infer", {}).get("iou", 0.5))
    imgsz= int(cfg.get("infer", {}).get("imgsz", 896))
    names = cfg.get("drone_class_names", ["drone"])

    imgs = sorted([p for p in glob.glob(os.path.join(in_dir, "*")) if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
    for p in tqdm(imgs, desc="RF-DETR infer"):
        img = cv2.imread(p)
        if img is None: continue
        boxes, scores, _ = predict_image(model, img, conf, iou, imgsz)
        out_path = os.path.join(out_dir, os.path.basename(p))
        draw_and_save(img.copy(), boxes, scores, names, out_path, thresh=conf)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    infer_dir(args.cfg, args.input_dir, args.out_dir)

if __name__ == "__main__":
    main()
