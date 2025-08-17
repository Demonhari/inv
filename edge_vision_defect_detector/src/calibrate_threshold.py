import os, yaml, numpy as np, cv2
from joblib import load
from tqdm import tqdm
from pycocotools.coco import COCO
from .candidate import detect_candidates
from .features import patch_features
from .utils import nms, xywh_to_xyxy, clip_box, box_iou, extract_patch

def detect_image(img, model, cfg, score_thresh):
    boxes, _ = detect_candidates(img,
        cfg["candidate"]["sigmas"], cfg["candidate"]["thresh_rel"],
        cfg["candidate"]["min_dist"], cfg["candidate"]["base_box"])
    feats, keep_idx = [], []
    for i,b in enumerate(boxes):
        p = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
        if p is None: continue
        feats.append(patch_features(p, model["hog"], model["lbp"]))
        keep_idx.append(i)
    if not feats: return [], []
    scores = model["pipe"].decision_function(np.stack(feats)).tolist()
    boxes = [boxes[i] for i in keep_idx]
    # threshold + NMS
    keep = [i for i,s in enumerate(scores) if s >= score_thresh]
    boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
    keep = nms(boxes, scores, cfg["nms_iou"])
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    coco = COCO(cfg["val_coco"])
    model = load(os.path.join(cfg["model_dir"], "hog_lbp_linear_svm.joblib"))
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    eval_iou = float(cfg["eval_iou"])
    img_ids = coco.getImgIds()

    thresh_grid = np.linspace(-2.0, 1.0, 31)
    best = (0.0, None)  # f1, thresh

    for th in thresh_grid:
        TP=FP=0; n_gt=0
        for img_id in img_ids:
            info = coco.loadImgs([img_id])[0]
            p = os.path.join(os.path.dirname(cfg["val_coco"]), info["file_name"])
            img = cv2.imread(p); 
            if img is None: continue
            H,W = img.shape[:2]
            gt = []
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
                if ann["category_id"] in drone_ids:
                    gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
            n_gt += len(gt)
            det_b, det_s = detect_image(img, model, cfg, th)
            matched = set()
            for b,_ in sorted(zip(det_b, det_s), key=lambda x:-x[1]):
                if not gt: 
                    FP += 1; continue
                ious = [box_iou(b, g) for g in gt]
                m = int(np.argmax(ious))
                if ious[m] >= eval_iou and m not in matched:
                    TP += 1; matched.add(m)
                else:
                    FP += 1
        prec = TP / max(1, TP+FP)
        rec  = TP / max(1, n_gt)
        f1 = (2*prec*rec)/(prec+rec+1e-9)
        if f1 > best[0]:
            best = (f1, th)
        print(f"th={th:.2f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    print(f"\nBest F1={best[0]:.3f} at score_thresh={best[1]:.2f}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/default.yaml")
