import os, argparse, yaml, random
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from .utils import xywh_to_xyxy, clip_box, extract_patch, ensure_dir

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    patch_size = int(cfg["patch_size"]); pad = int(cfg["pad"])
    neg_per_pos = int(cfg["train"]["neg_per_pos"])

    for split, coco_path in [("train", cfg["train_coco"]), ("val", cfg["val_coco"])]:
        coco = COCO(coco_path)
        class_ids = coco.getCatIds(catNms=cfg["drone_class_names"])
        ignore_ids = coco.getCatIds(catNms=cfg.get("ignore_class_names", []))
        img_ids = coco.getImgIds()

        pos_dir = os.path.join(cfg["patch_root"], f"{split}_pos"); ensure_dir(pos_dir)
        neg_dir = os.path.join(cfg["patch_root"], f"{split}_neg"); ensure_dir(neg_dir)

        for img_id in tqdm(img_ids, desc=f"Extract {split}"):
            img_info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(os.path.dirname(coco_path), img_info["file_name"])
            img = cv2.imread(img_path); 
            if img is None: continue
            H,W = img.shape[:2]
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            drone_boxes = [xywh_to_xyxy(a["bbox"]) for a in anns if a["category_id"] in class_ids]
            ignore_boxes = [xywh_to_xyxy(a["bbox"]) for a in anns if a["category_id"] in ignore_ids]

            # POSITIVES
            for i, b in enumerate(drone_boxes):
                bb = clip_box(b, W, H)
                patch = extract_patch(img, bb, patch_size=patch_size, pad=pad)
                if patch is None: continue
                cv2.imwrite(os.path.join(pos_dir, f"{img_id}_{i}.png"), patch)

            # EASY NEGATIVES
            n_easy = max(1, neg_per_pos*max(1, len(drone_boxes)))
            attempts = 0; written = 0
            while written < n_easy and attempts < n_easy*15:
                attempts += 1
                sz = random.randint(patch_size//2, patch_size*2)
                x1 = random.randint(0, max(0, W-sz)); y1 = random.randint(0, max(0, H-sz))
                x2, y2 = x1+sz, y1+sz
                candidate = [x1,y1,x2,y2]
                # reject overlaps with drones or ignore areas
                bad = False
                for b in drone_boxes+ignore_boxes:
                    # simple IoU check
                    bx1,by1,bx2,by2 = b
                    ix1,iy1 = max(x1,bx1), max(y1,by1)
                    ix2,iy2 = min(x2,bx2), min(y2,by2)
                    if ix2>ix1 and iy2>iy1:
                        inter = (ix2-ix1)*(iy2-iy1)
                        if inter/((x2-x1)*(y2-y1)+ (bx2-bx1)*(by2-by1) - inter +1e-9) > 0.05:
                            bad=True; break
                if bad: continue
                patch = extract_patch(img, candidate, patch_size=patch_size, pad=pad)
                if patch is None: continue
                cv2.imwrite(os.path.join(neg_dir, f"{img_id}_rand_{written}.png"), patch)
                written += 1

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/default.yaml")
