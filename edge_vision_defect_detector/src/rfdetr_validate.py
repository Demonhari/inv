import os, yaml, numpy as np, cv2, torch, numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from rfdetr import RFDETRBase
from .utils import box_iou, xywh_to_xyxy, clip_box

from src.rfdetr_infer import call_rfdetr, parse_predictions  # reuse robust helpers

def evaluate(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    coco = COCO(cfg["val_coco"])
    img_root = os.path.dirname(cfg["val_coco"])
    p = cfg["rfdetr"]
    model = RFDETRBase(model=p.get("variant","s"))
    conf  = float(p.get("conf",0.25))
    iou_p = float(p.get("iou",0.5))
    imgsz = int(p.get("imgsz",640))
    eval_iou = float(cfg.get("eval_iou",0.20))
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])

    TP=FP=0; n_gt=0
    for img_id in tqdm(coco.getImgIds(), desc="Validate(RF-DETR)"):
        info = coco.loadImgs([img_id])[0]
        pth  = os.path.join(img_root, info["file_name"])
        img  = cv2.imread(pth)
        if img is None: continue
        H,W  = img.shape[:2]

        gt=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                gt.append(clip_box(xywh_to_xyxy(ann["bbox"]), W, H))
        n_gt += len(gt)

        raw = call_rfdetr(model, img, conf, iou_p, imgsz)
        dets = parse_predictions(raw)

        matched=set()
        for (bb, sc, cls) in sorted(dets, key=lambda x: -x[1]):
            if not gt: FP+=1; continue
            ious = [box_iou(bb, g) for g in gt]
            m = int(np.argmax(ious)) if ious else -1
            if ious and ious[m] >= eval_iou and m not in matched:
                TP+=1; matched.add(m)
            else:
                FP+=1

    prec = TP/max(1,TP+FP); rec = TP/max(1,n_gt)
    print(f"GT: {n_gt} | P={prec:.3f} | R={rec:.3f}")

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv)>=2 else "configs/rfdetr.yaml"
    evaluate(cfg)
