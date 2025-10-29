# src/candidate.py
import cv2
import numpy as np
from typing import List, Tuple, Optional
from .utils import clip_box

def _peak_coords(resp: np.ndarray, thresh: float) -> List[Tuple[int, int, float]]:
    """Return (y, x, val) at 3x3 local maxima above thresh."""
    if resp.size == 0:
        return []
    dil = cv2.dilate(resp, np.ones((3, 3), np.uint8))
    msk = (resp == dil) & (resp >= thresh)
    ys, xs = np.where(msk)
    return [(int(y), int(x), float(resp[y, x])) for y, x in zip(ys, xs)]

def _greedy_nms_points(pts: List[Tuple[int, int, float]], min_dist: int) -> List[Tuple[int, int, float]]:
    """Greedy keep of points by descending score with L2 distance >= min_dist."""
    if not pts:
        return []
    pts = sorted(pts, key=lambda t: -t[2])
    kept: List[Tuple[int, int, float]] = []
    r2 = float(min_dist) * float(min_dist)
    for y, x, s in pts:
        ok = True
        for ky, kx, _ in kept:
            dy = float(y - ky)
            dx = float(x - kx)
            if dy * dy + dx * dx < r2:
                ok = False
                break
        if ok:
            kept.append((y, x, s))
    return kept

def detect_candidates(
    img: np.ndarray,
    sigmas: List[float],
    thresh_rel: float,
    min_dist: int,
    base_box: int,
    mask_top_ratio: Optional[float] = None,
    min_contrast: float = 0.0,
    mask_bottom_ratio: Optional[float] = None,
    max_local_std: Optional[float] = None,
) -> Tuple[List[List[int]], List[float]]:
    """
    Find small blob-like candidates and return (boxes, scores).

    boxes: [x1, y1, x2, y2] in image coords (clipped)
    scores: simple local-contrast score (higher = more distinct)
    """
    # --- prep gray + ROI masks ---
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray_u8.shape[:2]

    if mask_top_ratio is not None:
        t = max(0, min(H, int(H * float(mask_top_ratio))))
        gray_u8[:t, :] = 0

    if mask_bottom_ratio is not None:
        # ignore bottom mask_bottom_ratio fraction (set to 0 so no responses form)
        cutoff = max(0, min(H, int(H * (1.0 - float(mask_bottom_ratio)))))
        gray_u8[cutoff:, :] = 0

    gray = gray_u8.astype(np.float32)

    # --- multi-sigma LoG-ish responses + peak pick ---
    pts_all: List[Tuple[int, int, float, float]] = []  # (y, x, score, sigma)
    for sigma in sigmas:
        if sigma <= 0:
            continue
        g = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        lap = cv2.Laplacian(g, cv2.CV_32F)
        resp = np.abs(lap)
        m = float(resp.max())
        if m <= 0:
            continue
        thr = float(thresh_rel) * m
        peaks = _peak_coords(resp, thr)
        for (y, x, s) in peaks:
            pts_all.append((y, x, s, float(sigma)))

    if not pts_all:
        return [], []

    # merge all-sigma peaks with distance gating
    pts_kept = _greedy_nms_points([(y, x, s) for (y, x, s, _) in pts_all], int(min_dist))

    # --- build boxes + scoring & optional gates ---
    boxes: List[List[int]] = []
    scores: List[float] = []
    for (y, x, s) in pts_kept:
        # size from sigma & base_box
        # find closest sigma used for this point (optional; not strictly needed)
        # choose conservative half-size
        half = max(int(base_box // 2), 3)

        x1 = int(x - half)
        y1 = int(y - half)
        x2 = int(x + half)
        y2 = int(y + half)
        x1, y1, x2, y2 = clip_box([x1, y1, x2, y2], W, H)

        if x2 <= x1 or y2 <= y1:
            continue

        # local contrast score
        patch = gray_u8[y1:y2, x1:x2]
        sc = float(patch.mean() - patch.min()) if patch.size else 0.0

        # gate by contrast
        if sc < float(min_contrast):
            continue

        # optional local texture (std) gate around center
        if max_local_std is not None:
            k = max(2, half)
            xa, ya = max(0, x - k), max(0, y - k)
            xb, yb = min(W, x + k + 1), min(H, y + k + 1)
            win = gray_u8[ya:yb, xa:xb]
            if win.size and float(win.std()) > float(max_local_std):
                continue

        boxes.append([x1, y1, x2, y2])
        scores.append(sc)

    return boxes, [float(s) for s in scores]
