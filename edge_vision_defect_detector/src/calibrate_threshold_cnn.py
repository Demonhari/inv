import os
import yaml
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pycocotools.coco import COCO

from .utils import nms, xywh_to_xyxy, clip_box, box_iou, extract_patch
from .candidate import detect_candidates
from .models_cnn import build_model


def score_image(img, model, device, cfg):
    """
    Run candidate generation on a full image, extract CNN scores for each patch.
    Returns (boxes, scores) where boxes are [x1,y1,x2,y2] in image coords and
    scores are raw logits (floats) from the CNN (before sigmoid).
    """
    cand = cfg["candidate"]
    boxes, _ = detect_candidates(
        img,
        cand["sigmas"],
        cand["thresh_rel"],
        cand["min_dist"],
        cand["base_box"],
        cand.get("mask_top_ratio"),
        cand.get("min_contrast", 0.0),
    )

    feats = []
    keep_idx = []
    for i, b in enumerate(boxes):
        patch = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
        if patch is None:
            continue
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        feats.append(torch.from_numpy(g).float().div(255.0).unsqueeze(0).unsqueeze(0))
        keep_idx.append(i)

    if not feats:
        return [], []

    x = torch.cat(feats, 0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().tolist()

    boxes = [boxes[i] for i in keep_idx]
    scores = logits
    return boxes, scores


def main(cfg_path, arch):
    """
    Sweep the CNN score threshold on the validation set and print the best F1.
    Usage:
      python -m src.calibrate_threshold_cnn configs/cnn.yaml
      python -m src.calibrate_threshold_cnn configs/cnn.yaml xl
    """
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cpu")

    # load model
    ckpt_path = os.path.join(cfg["model_dir"], f"cnn_{arch}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # data / params
    coco = COCO(cfg["val_coco"])
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    eval_iou = float(cfg["eval_iou"])
    nms_iou = float(cfg["nms_iou"])
    max_k = int(cfg.get("max_dets_per_image", 100000))

    img_ids = coco.getImgIds()

    # precompute detections for all images once
    all_boxes = []
    all_scores = []
    per_img_gt = []

    for img_id in tqdm(img_ids, desc="Precompute"):
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(os.path.dirname(cfg["val_coco"]), info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            all_boxes.append([])
            all_scores.append([])
            per_img_gt.append([])
            continue

        H, W = img.shape[:2]

        gt = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        per_img_gt.append(gt)

        boxes, scores = score_image(img, model, device, cfg)
        all_boxes.append(boxes)
        all_scores.append(scores)

    best_f1 = 0.0
    best_th = None

    # sweep a broad range of thresholds; adjust if needed
    for th in np.linspace(-0.5, 0.8, 27):
        TP = FP = 0
        n_gt = sum(len(g) for g in per_img_gt)

        for boxes, scores, gt in zip(all_boxes, all_scores, per_img_gt):
            # threshold
            keep = [i for i, s in enumerate(scores) if s >= th]
            b = [boxes[i] for i in keep]
            s = [scores[i] for i in keep]

            # NMS
            if b:
                keep_nms = nms(b, s, nms_iou)
                b = [b[i] for i in keep_nms]
                s = [s[i] for i in keep_nms]

                # cap per-image detections
                if len(s) > max_k:
                    order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:max_k]
                    b = [b[i] for i in order]
                    s = [s[i] for i in order]

            # greedy match to GT
            matched = set()
            for bb, ss in sorted(zip(b, s), key=lambda x: -x[1]):
                if not gt:
                    FP += 1
                    continue
                ious = [box_iou(bb, g) for g in gt]
                if not ious:
                    FP += 1
                    continue
                m = int(np.argmax(ious))
                if ious[m] >= eval_iou and m not in matched:
                    TP += 1
                    matched.add(m)
                else:
                    FP += 1

        prec = TP / max(1, TP + FP)
        rec = TP / max(1, n_gt)
        f1 = (2 * prec * rec) / (prec + rec + 1e-9)
        print(f"th={th:.2f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\nBest F1={best_f1:.3f} at score_thresh={best_th:.2f}")


if __name__ == "__main__":
    import sys
    # usage:
    #   python -m src.calibrate_threshold_cnn configs/cnn.yaml
    #   python -m src.calibrate_threshold_cnn configs/cnn.yaml xl
    args = sys.argv[1:] or ["configs/cnn.yaml"]
    if len(args) >= 2:
        cfg_arg, arch_arg = args[0], args[1]
    else:
        cfg_arg, arch_arg = args[0], "large"
    main(cfg_arg, arch_arg)
