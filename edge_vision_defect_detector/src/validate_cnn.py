# src/validate_cnn.py
import os, yaml, numpy as np, cv2, torch
from tqdm import tqdm
from pycocotools.coco import COCO

from .utils import nms, xywh_to_xyxy, clip_box, box_iou, extract_patch
from .candidate import detect_candidates
from .models_cnn import build_model


# ---------- light blobness verifier (LoG) ----------
def _norm_log_response(gray, cx, cy, sigma):
    k = max(1, int(3 * sigma))
    H, W = gray.shape[:2]
    xa, ya = max(0, cx - k), max(0, cy - k)
    xb, yb = min(W, cx + k + 1), min(H, cy + k + 1)
    win = gray[ya:yb, xa:xb].astype(np.float32)
    if win.size == 0:
        return 0.0
    g = cv2.GaussianBlur(win, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    lap = cv2.Laplacian(g, cv2.CV_32F)
    denom = np.sqrt((g * g).mean() + 1e-6)
    return float(np.abs(lap).mean() / denom)


def blobness_ok(img, box, base_sigma=1.2, tau=0.22):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r1 = _norm_log_response(gray, cx, cy, base_sigma)
    r2 = _norm_log_response(gray, cx, cy, base_sigma * 1.4)
    return max(r1, r2) >= tau


# ---------- scoring ----------
def score_image(img, model, device, cfg):
    cand = cfg["candidate"]
    boxes, _ = detect_candidates(
        img,
        cand["sigmas"],
        cand["thresh_rel"],
        cand["min_dist"],
        cand["base_box"],
        cand.get("mask_top_ratio"),
        cand.get("min_contrast", 0.0),
        cand.get("mask_bottom_ratio"),
        cand.get("max_local_std"),
    )

    feats, idx = [], []
    for i, b in enumerate(boxes):
        p = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
        if p is None:
            continue
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        feats.append(torch.from_numpy(g).float().div(255.0).unsqueeze(0).unsqueeze(0))
        idx.append(i)

    if not feats:
        return [], []

    x = torch.cat(feats, 0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy().tolist()

    boxes = [boxes[i] for i in idx]
    scores = logits

    # post-score verifier
    v_boxes, v_scores = [], []
    base_sigma = float(cand.get("blob_sigma", 1.2))
    tau = float(cfg.get("blobness_tau", 0.22))
    for b, s in zip(boxes, scores):
        if blobness_ok(img, b, base_sigma=base_sigma, tau=tau):
            v_boxes.append(b)
            v_scores.append(s)
    return v_boxes, v_scores


# ---------- AP utilities ----------
def compute_ap(all_boxes, all_scores, all_gt, iou_thr, nms_iou, max_per_img):
    # flatten detections with image index
    dets = []
    for i, (b, s) in enumerate(zip(all_boxes, all_scores)):
        if not b:
            continue
        keep = nms(b, s, nms_iou)
        b = [b[k] for k in keep]
        s = [s[k] for k in keep]
        if len(s) > max_per_img:
            order = sorted(range(len(s)), key=lambda k: s[k], reverse=True)[:max_per_img]
            b = [b[k] for k in order]
            s = [s[k] for k in order]
        for bb, ss in zip(b, s):
            dets.append((i, float(ss), bb))

    # sort by score desc
    dets.sort(key=lambda x: -x[1])

    n_gt = sum(len(g) for g in all_gt)
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0, 0  # AP, max_prec, max_rec, TP, FP

    tp = np.zeros(len(dets), dtype=np.float32)
    fp = np.zeros(len(dets), dtype=np.float32)
    matched = [set() for _ in all_gt]

    for k, (img_i, score, bb) in enumerate(dets):
        gt = all_gt[img_i]
        if not gt:
            fp[k] = 1
            continue
        ious = [box_iou(bb, g) for g in gt]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and j not in matched[img_i]:
            tp[k] = 1
            matched[img_i].add(j)
        else:
            fp[k] = 1

    # precision-recall curve
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    prec = cum_tp / np.maximum(1, cum_tp + cum_fp)
    rec = cum_tp / max(1, n_gt)

    # AP = area under precision envelope
    # make precision non-increasing w.r.t recall
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # integrate
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    max_prec = float(prec.max()) if prec.size else 0.0
    max_rec = float(rec.max()) if rec.size else 0.0
    TP = int(cum_tp[-1]) if cum_tp.size else 0
    FP = int(cum_fp[-1]) if cum_fp.size else 0
    return ap, max_prec, max_rec, TP, FP


def main(cfg_path, arch="large"):
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device("cpu")
    ckpt = torch.load(os.path.join(cfg["model_dir"], f"cnn_{arch}.pt"), map_location=dev)
    model = build_model(ckpt["arch"]).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()

    coco = COCO(cfg["val_coco"])
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
    eval_iou = float(cfg["eval_iou"])
    nms_iou = float(cfg["nms_iou"])
    max_k = int(cfg.get("max_dets_per_image", 100000))

    img_ids = coco.getImgIds()
    all_boxes, all_scores, all_gt = [], [], []

    pbar = tqdm(img_ids, desc="Validate(CNN)")
    for img_id in pbar:
        info = coco.loadImgs([img_id])[0]
        p = os.path.join(os.path.dirname(cfg["val_coco"]), info["file_name"])
        img = cv2.imread(p)
        if img is None:
            all_boxes.append([])
            all_scores.append([])
            all_gt.append([])
            continue

        H, W = img.shape[:2]
        gt = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        all_gt.append(gt)

        b, s = score_image(img, model, dev, cfg)
        all_boxes.append(b)
        all_scores.append(s)

    ap, max_prec, max_rec, TP, FP = compute_ap(
        all_boxes, all_scores, all_gt, eval_iou, nms_iou, max_k
    )

    n_gt = sum(len(g) for g in all_gt)
    total_dets = TP + FP
    print(f"GT: {n_gt} | Dets: {total_dets} | AP@{eval_iou:.2f} = {ap:.3f}")
    print(f"Max Prec: {max_prec:.3f} | Max Rec: {max_rec:.3f}")


if __name__ == "__main__":
    import sys
    # usage:
    #   python -m src.validate_cnn configs/cnn.yaml
    #   python -m src.validate_cnn --arch xl configs/cnn.yaml
    args = sys.argv[1:] or ["configs/cnn.yaml"]
    if len(args) >= 3 and args[0] in ("--arch", "-a"):
        arch = args[1]
        cfg = args[2]
    else:
        arch = "large"
        cfg = args[0]
    main(cfg, arch)
