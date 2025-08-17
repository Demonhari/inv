import os, yaml, numpy as np, cv2
from joblib import load
from tqdm import tqdm
from pycocotools.coco import COCO
from .candidate import detect_candidates
from .features import patch_features
from .utils import nms, xywh_to_xyxy, clip_box, extract_patch

def centers(boxes):
    return np.array([((x1+x2)/2.0, (y1+y2)/2.0) for x1,y1,x2,y2 in boxes], dtype=np.float32) if boxes else np.zeros((0,2), np.float32)

def detect_image(img, model, cfg, score_thresh=None):
    cand = cfg["candidate"]
    boxes, _ = detect_candidates(
        img,
        cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
        cand.get("mask_top_ratio"), cand.get("min_contrast", 0.0)
    )
    feats, keep_idx = [], []
    for i,b in enumerate(boxes):
        p = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
        if p is None: continue
        feats.append(patch_features(p, model["hog"], model["lbp"]))
        keep_idx.append(i)
    if not feats: return [], []
    scores = model["pipe"].decision_function(np.stack(feats)).tolist()
    boxes = [boxes[i] for i in keep_idx]
    st = cfg.get("score_thresh", 0.0) if score_thresh is None else score_thresh
    keep = [i for i,s in enumerate(scores) if s >= st]
    boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
    keep = nms(boxes, scores, cfg["nms_iou"])
    boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
    K = int(cfg.get('max_dets_per_image', 100000))
    if len(scores)>K:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
        boxes = [boxes[i] for i in order]; scores = [scores[i] for i in order]
    return boxes, scores

def evaluate_center(coco_path, model_path, cfg):
    coco = COCO(coco_path)
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    model = load(model_path)
    img_ids = coco.getImgIds()
    tol = float(cfg.get("center_tol", 6))

    TP=FP=0; n_gt=0
    for img_id in tqdm(img_ids, desc="Eval(center)"):
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(os.path.dirname(coco_path), info["file_name"])
        img = cv2.imread(img_path); 
        if img is None: continue
        H,W = img.shape[:2]
        gt_boxes=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt_boxes.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        n_gt += len(gt_boxes)
        det_boxes, det_scores = detect_image(img, model, cfg)
        if not gt_boxes:
            FP += len(det_boxes)
            continue
        gt_c = centers(gt_boxes)
        det_c = centers(det_boxes)
        used=set()
        for i,(cx,cy) in enumerate(det_c):
            d2 = np.sum((gt_c - np.array([cx,cy]))**2, axis=1)
            j = int(np.argmin(d2)) if len(d2) else -1
            if j>=0 and np.sqrt(d2[j]) <= tol and j not in used:
                TP += 1; used.add(j)
            else:
                FP += 1
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, n_gt)
    print(f"Center@{tol:.1f}px  P={prec:.3f} R={rec:.3f}  (TP={TP}, FP={FP}, GT={n_gt})")

if __name__ == "__main__":
    import sys, os
    cfg_path = sys.argv[1] if len(sys.argv)>1 else "configs/default.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    model_path = os.path.join(cfg["model_dir"], "hog_lbp_linear_svm.joblib")
    evaluate_center(cfg["val_coco"], model_path, cfg)
