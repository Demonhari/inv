import os, argparse, yaml, cv2, numpy as np
from joblib import load
from .candidate import detect_candidates
from .features import patch_features
from .utils import extract_patch, nms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--source", required=True, help="path to video file or 0 for webcam")
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    model = load(os.path.join(cfg["model_dir"], "hog_lbp_linear_svm.joblib"))
    src = 0 if args.source=="0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Cannot open source"); return
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    cand = cfg["candidate"]
    while True:
        ok, frame = cap.read()
        if not ok: break
        fg = bg.apply(frame)
        motion = cv2.medianBlur(fg, 5)
        motion = cv2.threshold(motion, 127, 255, cv2.THRESH_BINARY)[1]

        boxes, scores = detect_candidates(
            frame,
            cand["sigmas"], cand["thresh_rel"], cand["min_dist"], cand["base_box"],
            cand.get("mask_top_ratio"), cand.get("min_contrast", 0.0)
        )
        H,W = frame.shape[:2]
        filt_boxes, feats = [], []
        for b in boxes:
            x1,y1,x2,y2 = [int(v) for v in b]
            mroi = motion[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
            if mroi.size==0 or np.count_nonzero(mroi)==0: 
                continue
            p = extract_patch(frame, b, patch_size=cfg["patch_size"], pad=cfg["pad"])
            if p is None: continue
            feats.append(patch_features(p, model["hog"], model["lbp"]))
            filt_boxes.append(b)
        if feats:
            sc = model["pipe"].decision_function(np.stack(feats)).tolist()
            boxes = filt_boxes; scores = sc
            keep = [i for i,s in enumerate(scores) if s >= cfg.get("score_thresh", 0.0)]
            boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
            keep = nms(boxes, scores, cfg["nms_iou"])
            boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
            K = int(cfg.get('max_dets_per_image', 100000))
            if len(scores)>K:
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:K]
                boxes = [boxes[i] for i in order]; scores = [scores[i] for i in order]
        else:
            boxes, scores = [], []
        for b,s in zip(boxes, scores):
            x1,y1,x2,y2 = map(int,b)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.putText(frame,f"{s:.2f}",(x1,max(0,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,0),1,cv2.LINE_AA)
        if args.display:
            cv2.imshow("detections", frame)
            if cv2.waitKey(1)==27: break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
