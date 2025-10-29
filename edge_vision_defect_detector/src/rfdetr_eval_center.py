import os, yaml, numpy as np, cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from rfdetr import RFDETRBase
from src.rfdetr_infer import call_rfdetr, parse_predictions

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    coco = COCO(cfg["val_coco"])
    img_root = os.path.dirname(cfg["val_coco"])
    p = cfg["rfdetr"]
    model = RFDETRBase(model=p.get("variant","s"))
    conf  = float(p.get("conf",0.25))
    iou_p = float(p.get("iou",0.5))
    imgsz = int(p.get("imgsz",640))
    tol   = float(cfg.get("center_tol",6))
    drone_ids = coco.getCatIds(catNms=cfg["drone_class_names"])

    TP=FP=0; n_gt=0
    for img_id in tqdm(coco.getImgIds(), desc="Eval(center,RF-DETR)"):
        info = coco.loadImgs([img_id])[0]
        pth  = os.path.join(img_root, info["file_name"])
        img  = cv2.imread(pth)
        if img is None: continue
        H,W  = img.shape[:2]

        gt_c=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann["category_id"] in drone_ids:
                x,y,w,h = ann["bbox"]; gt_c.append([x+w/2, y+h/2])
        n_gt += len(gt_c)

        raw = call_rfdetr(model, img, conf, iou_p, imgsz)
        dets = parse_predictions(raw)
        det_c = [((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for (b,_,_) in dets]

        used=set()
        for cx,cy in det_c:
            if not gt_c: FP+=1; continue
            d2 = np.sum((np.array(gt_c) - np.array([cx,cy]))**2, axis=1)
            j = int(np.argmin(d2))
            if np.sqrt(d2[j]) <= tol and j not in used:
                TP += 1; used.add(j)
            else:
                FP += 1

    prec = TP/max(1,TP+FP); rec=TP/max(1,n_gt)
    print(f"Center@{tol:.1f}px  P={prec:.3f} R={rec:.3f}  (TP={TP}, FP={FP}, GT={n_gt})")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>=2 else "configs/rfdetr.yaml")
