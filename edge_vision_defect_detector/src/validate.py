import os, yaml, math
import numpy as np
import cv2
from tqdm import tqdm
from joblib import load
from pycocotools.coco import COCO
from .utils import nms, xywh_to_xyxy, clip_box, box_iou, extract_patch, ensure_dir
from .candidate import detect_candidates
from .features import patch_features

def detect_image(img, model, cfg):
    cand = cfg["candidate"]
    boxes, _ = detect_candidates(
        img,
        cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
        cand.get("mask_top_ratio"), cand.get("min_contrast", 0.0)
    )
    feats = []
    for b in boxes:
        p = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
        feats.append(patch_features(p, model["hog"], model["lbp"]) if p is not None else None)
    feats = [f for f in feats if f is not None]
    if len(feats)==0: return [], []
    pred = model["pipe"].decision_function(np.stack(feats)).tolist()

    # threshold
    keep = [i for i,s in enumerate(pred) if s >= cfg.get("score_thresh", 0.0)]
    boxes = [boxes[i] for i in keep]; scores = [pred[i] for i in keep]
    # NMS
    keep_idx = nms(boxes, scores, cfg["nms_iou"])
    boxes = [boxes[i] for i in keep_idx]; scores = [scores[i] for i in keep_idx]
    # top-K
    K = int(cfg.get("max_dets_per_image", 100000))
    if len(scores) > K:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
        boxes = [boxes[i] for i in order]; scores = [scores[i] for i in order]
    return boxes, scores

def evaluate(coco_path, model_path, cfg):
    coco = COCO(coco_path)
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    eval_iou = float(cfg["eval_iou"])
    model = load(model_path)

    img_ids = coco.getImgIds()
    TP, FP, scores_all = [], [], []
    n_gt = 0
    for img_id in tqdm(img_ids, desc="Validate"):
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(os.path.dirname(coco_path), info["file_name"])
        img = cv2.imread(img_path); 
        if img is None: continue
        H,W = img.shape[:2]
        gt = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        n_gt += len(gt)
        det_boxes, det_scores = detect_image(img, model, cfg)
        matched = set()
        for b,s in sorted(zip(det_boxes, det_scores), key=lambda x: -x[1]):
            ious = [box_iou(b, g) for g in gt]
            m = int(np.argmax(ious)) if ious else -1
            if ious and ious[m] >= eval_iou and m not in matched:
                TP.append(1); FP.append(0); matched.add(m)
            else:
                TP.append(0); FP.append(1)
            scores_all.append(s)
    if len(scores_all)==0:
        print("No detections."); return
    order = np.argsort(-np.array(scores_all))
    TP = np.array(TP)[order]; FP = np.array(FP)[order]
    cumTP, cumFP = np.cumsum(TP), np.cumsum(FP)
    rec = cumTP / max(1, n_gt)
    prec = cumTP / np.maximum(1, cumTP+cumFP)
    ap = 0.0
    for t in np.linspace(0,1,11):
        p = np.max(prec[rec>=t]) if np.any(rec>=t) else 0
        ap += p/11
    print(f"GT: {n_gt} | Dets: {len(scores_all)} | AP@{eval_iou:.2f} = {ap:.3f}")
    print(f"Max Prec: {prec.max():.3f} | Max Rec: {rec.max():.3f}")

if __name__ == "__main__":
    import sys, os
    cfg_path = sys.argv[1] if len(sys.argv)>1 else "configs/default.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    model_path = os.path.join(cfg["model_dir"], "hog_lbp_linear_svm.joblib")
    evaluate(cfg["val_coco"], model_path, cfg)
