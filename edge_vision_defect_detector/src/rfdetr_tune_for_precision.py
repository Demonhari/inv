import os, yaml, numpy as np, cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from rfdetr import RFDETRBase
from .utils import box_iou, xywh_to_xyxy, clip_box

def call_rfdetr(model, img, conf, iou, imgsz):
    for fn in ("predict","infer","__call__"):
        if hasattr(model, fn):
            return getattr(model, fn)(
                img, confidence=conf, iou_threshold=iou, imgsz=imgsz
            ) if fn!="__call__" else model(img, confidence=conf, iou_threshold=iou, imgsz=imgsz)
    raise AttributeError("RFDETR model has no predict/infer/__call__")

def parse_preds(out):
    preds=[]
    if hasattr(out,"predictions"):
        for p in out.predictions:
            xyxy = getattr(p,"xyxy",[p.x1,p.y1,p.x2,p.y2])
            conf = float(getattr(p,"confidence", getattr(p,"conf",0.0)))
            preds.append((list(map(float,xyxy)), conf))
        return preds
    if isinstance(out, dict) and "predictions" in out:
        for p in out["predictions"]:
            xyxy = p.get("xyxy") or [p["x1"],p["y1"],p["x2"],p["y2"]]
            conf = float(p.get("confidence", p.get("conf",0.0)))
            preds.append((list(map(float,xyxy)), conf))
        return preds
    if isinstance(out,(list,tuple)):
        for p in out:
            if isinstance(p, dict):
                xyxy = p.get("xyxy") or [p["x1"],p["y1"],p["x2"],p["y2"]]
                conf = float(p.get("confidence", p.get("conf",0.0)))
                preds.append((list(map(float,xyxy)), conf))
        return preds
    raise RuntimeError("Unrecognized RF-DETR output format")

def main(cfg_path, target_prec=0.90):
    cfg = yaml.safe_load(open(cfg_path))
    coco = COCO(cfg["val_coco"])
    img_root = os.path.dirname(cfg["val_coco"])
    p = cfg["rfdetr"]
    model = RFDETRBase(model=p.get("variant","s"))
    iou_post = float(p.get("iou",0.5))
    imgsz    = int(p.get("imgsz",640))
    eval_iou = float(cfg.get("eval_iou",0.20))
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])

    all_preds=[]; all_gt=[]
    for img_id in tqdm(coco.getImgIds(), desc="Precompute"):
        info = coco.loadImgs([img_id])[0]
        pth  = os.path.join(img_root, info["file_name"])
        img  = cv2.imread(pth)
        if img is None: all_preds.append([]); all_gt.append([]); continue
        H,W  = img.shape[:2]
        gt=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        all_gt.append(gt)
        raw = call_rfdetr(model, img, conf=0.001, iou=iou_post, imgsz=imgsz)
        all_preds.append(parse_preds(raw))

    best={"conf":None, "prec":0.0, "rec":0.0}
    for th in np.linspace(0.05, 0.95, 19):
        TP=FP=0; n_gt=sum(len(g) for g in all_gt)
        for preds, gt in zip(all_preds, all_gt):
            filt = [(b,s) for (b,s) in preds if s >= th]
            matched=set()
            for (bb, s) in sorted(filt, key=lambda x: -x[1]):
                if not gt: FP+=1; continue
                ious=[box_iou(bb, g) for g in gt]
                m = int(np.argmax(ious)) if ious else -1
                if ious and ious[m] >= eval_iou and m not in matched:
                    TP+=1; matched.add(m)
                else:
                    FP+=1
        prec = TP/max(1,TP+FP); rec=TP/max(1,n_gt)
        print(f"conf={th:.2f}  P={prec:.3f} R={rec:.3f}")
        if prec >= target_prec and rec >= best["rec"]:
            best={"conf":float(th), "prec":float(prec), "rec":float(rec)}

    if best["conf"] is None:
        print("\nNo confidence reached the target precision. You can raise it manually in configs/rfdetr.yaml['rfdetr']['conf'].")
    else:
        print(f"\nChosen conf for precision ≥ {target_prec:.2f}: {best['conf']:.2f}")
        print(f"Estimated Val: P={best['prec']:.3f}, R={best['rec']:.3f}")
        cfg["rfdetr"]["conf"] = best["conf"]
        open(cfg_path,'w').write(yaml.dump(cfg, sort_keys=False))
        print(f"Updated conf in {cfg_path}")

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv)>=2 else "configs/rfdetr.yaml"
    main(cfg, target_prec=0.90)
