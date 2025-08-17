import os, argparse, yaml, glob
import numpy as np
import cv2
from joblib import load
from .candidate import detect_candidates
import numpy as np
from .features import patch_features
from .utils import extract_patch, nms, ensure_dir

def draw_boxes(img, boxes, scores):
    for b,s in zip(boxes, scores):
        x1,y1,x2,y2 = map(int,b)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.putText(img, f"{s:.2f}", (x1, max(0,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1, cv2.LINE_AA)
    return img

def run_on_path(path, out_dir, model, cfg):
    ims = []
    if os.path.isdir(path):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp")
        for e in exts: ims.extend(sorted(glob.glob(os.path.join(path,e))))
    else:
        ims = [path]
    ensure_dir(out_dir)
    cand = cfg["candidate"]
    for p in ims:
        img = cv2.imread(p); 
        if img is None: continue
        boxes, scores = detect_candidates(
            img,
            cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
            cand.get("mask_top_ratio"), cand.get("min_contrast", 0.0)
        )
        feats, keep_idx = [], []
        for i,b in enumerate(boxes):
            patch = extract_patch(img, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
            if patch is None: continue
            feats.append(patch_features(patch, model["hog"], model["lbp"]))
            keep_idx.append(i)
        if feats:
            sc = model["pipe"].decision_function(np.stack(feats)).tolist()
            boxes = [boxes[i] for i in keep_idx]; scores = sc
            filt = [i for i,s in enumerate(scores) if s >= cfg.get("score_thresh", 0.0)]
            boxes = [boxes[i] for i in filt]; scores = [scores[i] for i in filt]
            keep = nms(boxes, scores, cfg["nms_iou"])
            boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
            K = int(cfg.get('max_dets_per_image', 100000))
            if len(scores)>K:
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
                boxes = [boxes[i] for i in order]; scores = [scores[i] for i in order]
        vis = draw_boxes(img.copy(), boxes, scores)
        outp = os.path.join(out_dir, os.path.basename(p))
        cv2.imwrite(outp, vis)
        print(f"Wrote {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="data/runs/infer_images")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    model = load(os.path.join(cfg["model_dir"], "hog_lbp_linear_svm.joblib"))
    run_on_path(args.input, args.out, model, cfg)

if __name__ == "__main__":
    main()
