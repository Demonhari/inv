#!/usr/bin/env python3
import argparse, json, random, shutil, glob, math, os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

# --------------------------
# COCO -> YOLO conversion
# --------------------------
def build_image_index(root: Path):
    idx = defaultdict(list)
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.JPG","*.PNG")
    for ext in exts:
        for p in root.rglob(ext):
            idx[p.name].append(p)
    return idx

def resolve_image_path(file_name: str, data_root: Path, img_index):
    p = Path(file_name)
    if p.is_file(): return p
    p2 = data_root / file_name
    if p2.is_file(): return p2
    matches = img_index.get(Path(file_name).name, [])
    if matches:
        matches = sorted(matches, key=lambda x: len(str(x)))
        return matches[0]
    return None

def coco_to_yolo_split(coco_path, split_name, yolo_img_dir, yolo_lbl_dir, data_root: Path):
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        coco = json.load(f)
    assert set(["images","annotations","categories"]).issubset(coco.keys()), "Not COCO detection JSON"

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    names = [cat_id_to_name[k] for k in sorted(cat_id_to_name.keys())]
    print(f"\n[{split_name}] categories in JSON:", names)

    id_to_img = {im["id"]: im for im in coco["images"]}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1: 
            continue
        anns_by_img[ann["image_id"]].append(ann)

    # Force desired ordering if present
    target_names = []
    for want in ["drones","ground"]:
        if want in names: target_names.append(want)
    for n in names:
        if n not in target_names: target_names.append(n)
    name_to_yolo = {n:i for i,n in enumerate(target_names)}
    print(f"[{split_name}] using classes:", target_names)

    img_index = build_image_index(data_root)
    written, missing = 0, 0

    for img_id, img in id_to_img.items():
        src_path = resolve_image_path(img["file_name"], data_root, img_index)
        if src_path is None or not Path(src_path).is_file():
            missing += 1
            continue

        dst_img = yolo_img_dir / Path(src_path).name
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)

        H, W = img["height"], img["width"]
        lbl_lines = []
        for ann in anns_by_img.get(img_id, []):
            catname = cat_id_to_name[ann["category_id"]]
            if catname not in name_to_yolo:
                continue
            cls = name_to_yolo[catname]
            x, y, w, h = ann["bbox"]
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lbl_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_lbl = yolo_lbl_dir / (dst_img.stem + ".txt")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lbl_lines))
        written += 1

    print(f"[{split_name}] wrote {written} images, {missing} missing (unresolved file names).")
    return target_names

def convert_coco_to_yolo(data_root: Path, work_dir: Path):
    random.seed(42)
    yolo_dir = work_dir / "yolo_drone_dataset"
    img_tr = yolo_dir / "images" / "train"
    img_va = yolo_dir / "images" / "val"
    lbl_tr = yolo_dir / "labels" / "train"
    lbl_va = yolo_dir / "labels" / "val"
    for d in [img_tr,img_va,lbl_tr,lbl_va]:
        d.mkdir(parents=True, exist_ok=True)
    data_yaml = work_dir / "drone_dataset.yaml"

    coco_jsons = sorted(glob.glob(str(data_root / "**/*.json"), recursive=True))
    assert coco_jsons, f"No COCO json files under {data_root}"
    print("Found JSON files:")
    for j in coco_jsons: print("  -", j)

    def pick_split(paths):
        tr, va = None, None
        for p in paths:
            n = p.lower()
            if "train" in n and tr is None: tr = p
            if "val"   in n and va is None: va = p
        if tr or va: return tr, va
        return paths[0], None

    train_json, val_json = pick_split(coco_jsons)
    print("\nChosen JSON(s):")
    print("  train_json:", train_json)
    print("  val_json  :", val_json)

    if val_json is None:
        with open(train_json,"r") as f: coco_all = json.load(f)
        img_ids = [im["id"] for im in coco_all["images"]]
        random.shuffle(img_ids)
        split_at = max(1, int(0.8 * len(img_ids)))
        train_ids, val_ids = set(img_ids[:split_at]), set(img_ids[split_at:])
        def write_split_json(ids_set, out_path):
            imgs = [im for im in coco_all["images"] if im["id"] in ids_set]
            ann  = [a for a in coco_all["annotations"] if a["image_id"] in ids_set]
            out = {"images": imgs, "annotations": ann, "categories": coco_all["categories"]}
            with open(out_path, "w") as f: json.dump(out, f)
        tmp_train = work_dir / "coco_train_split.json"
        tmp_val   = work_dir / "coco_val_split.json"
        write_split_json(train_ids, tmp_train)
        write_split_json(val_ids, tmp_val)
        train_json, val_json = str(tmp_train), str(tmp_val)

    names_tr = coco_to_yolo_split(train_json, "train", img_tr, lbl_tr, data_root)
    names_va = coco_to_yolo_split(val_json  , "val"  , img_va, lbl_va, data_root)

    names = names_tr[:]
    for n in names_va:
        if n not in names: names.append(n)
    print("\nFinal class list:", names)

    yaml = f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n"
    for i,n in enumerate(names):
        yaml += f"  {i}: {n}\n"
    with open(data_yaml, "w") as f: f.write(yaml)
    print("\nWrote", data_yaml)
    return yolo_dir, data_yaml

# --------------------------
# Training (RT-DETR -> fallback YOLOv8)
# --------------------------
def train_detector(data_yaml: Path, work_dir: Path, epochs=60, imgsz=1280, run_name="rf_detr_medium_drone"):
    os.environ.setdefault("WANDB_MODE","disabled")  # keep logging simple on a VM
    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    runs_dir = work_dir / "runs"

    def train_with(weights):
        print(f"\n--- Training with weights: {weights} ---")
        model = YOLO(weights)
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=-1,
            device=device,
            project="drone_detection",   # wandb-safe name
            name=run_name,
            save_dir=str(runs_dir),
            optimizer="SGD",
            cos_lr=True,
            close_mosaic=10,
            cache=False,
            patience=20,
            workers=8 if device != "cpu" else 0,
        )
        return model

    try:
        model = train_with("rtdetr-l.pt")
    except Exception as e:
        print("\n[Warning] RT-DETR failed:", repr(e))
        print("Falling back to YOLOv8-L.")
        model = train_with("yolov8l.pt")

    best = Path(model.trainer.best)
    print("\nBest weights:", best)
    return best, runs_dir / run_name

# --------------------------
# Sanity check (optional)
# --------------------------
def sanity_preview(best_weights: Path, work_dir: Path, imgsz=1280):
    from ultralytics import YOLO
    pred_dir = work_dir / "runs" / "debug_one"
    pred_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(best_weights))
    val_imgs = list((work_dir/"yolo_drone_dataset/images/val").glob("*.*"))
    if not val_imgs:
        print("No images in val split to preview."); return
    res = detector.predict(source=str(val_imgs[0]), imgsz=imgsz, conf=0.15, save=True,
                           project=str(work_dir/"runs"), name="debug_one", verbose=False)
    print("Wrote preview to:", pred_dir)

# --------------------------
# CV helper (tiny dots) + NMS
# --------------------------
def detect_blobs_small_dots(gray, min_area=3, max_area=400, ksize_small=3, ksize_large=11, thresh=0):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g1 = cv2.GaussianBlur(g, (ksize_small, ksize_small), 0)
    g2 = cv2.GaussianBlur(g, (ksize_large, ksize_large), 0)
    dog = cv2.absdiff(g1, g2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    if thresh == 0:
        _, bw = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area: 
            continue
        x,y,w,h = cv2.boundingRect(c)
        score = min(1.0, area/float(max_area))
        out.append((x,y,x+w,y+h, score))
    return out

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter == 0: return 0.0
    a_area = max(0, ax2-ax1) * max(0, ay2-ay1)
    b_area = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (a_area + b_area - inter + 1e-6)

def nms_boxes(boxes, scores, iou_thr=0.4, top_k=300):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < top_k:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < iou_thr]
    return keep

# --------------------------
# Video inference (DL + CV fusion)
# --------------------------
def infer_videos(best_weights: Path, work_dir: Path, video_dir: Path, infer_dir: Path, imgsz=1280, conf=0.15):
    from ultralytics import YOLO
    infer_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(best_weights))

    # "drones" class index
    names = detector.names if isinstance(detector.names, dict) else {i:n for i,n in enumerate(detector.names)}
    name_to_idx = {v:k for k,v in names.items()}
    drone_cls = name_to_idx.get("drones", 0)

    def draw_box(img, box, color, label):
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1- th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    def process_video(src_path: Path):
        cap = cv2.VideoCapture(str(src_path))
        assert cap.isOpened(), f"cannot open {src_path}"
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_path = infer_dir / (src_path.stem + "_inferenced.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        print(f"[{src_path.name}] {W}x{H} @ {FPS}fps | frames: {total if total else '?'}")

        while True:
            ok, frame = cap.read()
            if not ok: break

            # DL
            dl_boxes, dl_scores = [], []
            r = detector.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if int(c) == drone_cls:
                        dl_boxes.append(b.tolist())
                        dl_scores.append(float(s))

            # CV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = math.sqrt(W*H) / 1000.0
            minA = max(3, int(2 * scale))
            maxA = int(450 * scale)
            cv_boxes_scores = detect_blobs_small_dots(gray, min_area=minA, max_area=maxA)
            cv_boxes, cv_scores = [], []
            for (x1,y1,x2,y2,s) in cv_boxes_scores:
                pad = 2 + int(2*scale)
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(W-1, x2+pad), min(H-1, y2+pad)
                cv_boxes.append([x1,y1,x2,y2])
                cv_scores.append(0.35 + 0.65*s)

            # Fuse
            all_boxes = np.array(dl_boxes + cv_boxes, dtype=float) if (dl_boxes or cv_boxes) else np.zeros((0,4))
            all_scores = np.array(dl_scores + cv_scores, dtype=float) if (dl_scores or cv_scores) else np.zeros((0,))
            src = (["DL"] * len(dl_boxes)) + (["CV"] * len(cv_boxes))
            if len(all_boxes) > 0:
                keep = nms_boxes(all_boxes, all_scores, iou_thr=0.4)
                for i in keep:
                    color = (0,255,0) if src[i]=="DL" else (0,255,255)
                    draw_box(frame, all_boxes[i], color, f"{src[i]}:{all_scores[i]:.2f}")

            writer.write(frame)

        writer.release()
        cap.release()
        print("Saved:", out_path)

    exts = ("*.mp4","*.mov","*.avi","*.mkv","*.MP4","*.MOV","*.AVI","*.MKV")
    video_files = []
    for e in exts: video_files.extend(sorted(video_dir.glob(e)))
    print(f"Found {len(video_files)} video(s) in {video_dir}")
    for v in video_files: process_video(v)

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Drone detection pipeline (VM)")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--work_dir" , type=Path, required=True)
    ap.add_argument("--video_dir", type=Path, required=False)
    ap.add_argument("--infer_dir", type=Path, required=False)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--run_name", type=str, default="rf_detr_medium_drone")
    ap.add_argument("--stage", choices=["convert","train","sanity","infer","all"], default="all")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ["convert","all"]:
        yolo_dir, data_yaml = convert_coco_to_yolo(args.data_root, args.work_dir)
    else:
        data_yaml = args.work_dir / "drone_dataset.yaml"
        assert data_yaml.exists(), f"{data_yaml} not found. Run --stage convert first."

    if args.stage in ["train","all"]:
        best, _ = train_detector(data_yaml, args.work_dir, epochs=args.epochs, imgsz=args.imgsz, run_name=args.run_name)
    else:
        # try to find best.pt under runs
        runs = args.work_dir / "runs" / args.run_name
        cands = sorted(runs.rglob("best*.pt"))
        if not cands:
            cands = sorted(runs.rglob("last*.pt"))
        assert cands, "No trained weights found. Run --stage train"
        best = cands[-1]

    if args.stage in ["sanity","all"]:
        sanity_preview(best, args.work_dir, imgsz=args.imgsz)

    if args.stage in ["infer","all"]:
        assert args.video_dir and args.infer_dir, "--video_dir and --infer_dir are required for inference"
        infer_videos(best, args.work_dir, args.video_dir, args.infer_dir, imgsz=args.imgsz, conf=0.15)

if __name__ == "__main__":
    main()
