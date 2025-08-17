import json, os, math
from typing import List, Tuple, Dict
import numpy as np
import cv2

def load_coco(json_path: str):
    with open(json_path, "r") as f:
        coco = json.load(f)
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    return coco, id_to_img, id_to_cat, anns_by_img

def find_cat_ids_by_name(coco, names: List[str]) -> List[int]:
    name_set = set(n.lower() for n in names)
    return [c["id"] for c in coco["categories"] if c["name"].lower() in name_set]

def xywh_to_xyxy(box):
    x,y,w,h = box
    return [x, y, x+w, y+h]

def clip_box(b, w, h):
    x1,y1,x2,y2 = b
    return [max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)]

def box_iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw*ih
    if inter == 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

def nms(boxes: List[List[float]], scores: List[float], iou_thresh: float):
    if len(boxes)==0: return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in order[1:]], dtype=np.float32)
        inds = np.where(ious <= iou_thresh)[0]
        order = order[inds+1]
    return keep

def extract_patch(img, box, patch_size=32, pad=8):
    h, w = img.shape[:2]
    x1,y1,x2,y2 = [int(v) for v in box]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half = patch_size//2
    # expand with pad
    size = patch_size + 2*pad
    x1 = int(cx - size/2); y1 = int(cy - size/2)
    x2 = int(cx + size/2); y2 = int(cy + size/2)
    x1,y1,x2,y2 = clip_box([x1,y1,x2,y2], w, h)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return crop

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
