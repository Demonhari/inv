import os, argparse, yaml, glob
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from joblib import dump
from .features import patch_features

def load_patches(dir_pos, dir_neg, hog_cfg, lbp_cfg, max_per=None):
    X, y = [], []
    pos = sorted(glob.glob(os.path.join(dir_pos, "*.png")))
    neg = sorted(glob.glob(os.path.join(dir_neg, "*.png")))
    if max_per:
        pos = pos[:max_per]; neg = neg[:max_per]
    for p in tqdm(pos, desc="Pos"):
        im = cv2.imread(p); 
        if im is None: continue
        X.append(patch_features(im, hog_cfg, lbp_cfg)); y.append(1)
    for p in tqdm(neg, desc="Neg"):
        im = cv2.imread(p); 
        if im is None: continue
        X.append(patch_features(im, hog_cfg, lbp_cfg)); y.append(0)
    return np.stack(X), np.array(y, dtype=np.int32)

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    hog_cfg = cfg["hog"]; lbp_cfg = cfg["lbp"]
    dtr = os.path.join(cfg["patch_root"], "train_pos"); dtn = os.path.join(cfg["patch_root"], "train_neg")
    dvr = os.path.join(cfg["patch_root"], "val_pos");   dvn = os.path.join(cfg["patch_root"], "val_neg")

    print("[*] Loading patches...")
    Xtr, ytr = load_patches(dtr, dtn, hog_cfg, lbp_cfg)
    Xva, yva = load_patches(dvr, dvn, hog_cfg, lbp_cfg) if os.path.isdir(dvr) else (None,None)

    print("[*] Training LinearSVC...")
    svc = LinearSVC(C=float(cfg["train"]["C"]), class_weight=cfg["train"]["class_weight"],
                    max_iter=int(cfg["train"]["max_iter"]), dual=True)
    clf = make_pipeline(StandardScaler(with_mean=False), svc)
    clf.fit(Xtr, ytr)
    if Xva is not None:
        yhat = clf.predict(Xva)
        print(classification_report(yva, yhat, digits=3))
    os.makedirs(cfg["model_dir"], exist_ok=True)
    out = os.path.join(cfg["model_dir"], "hog_lbp_linear_svm.joblib")
    dump({"pipe": clf, "hog": hog_cfg, "lbp": lbp_cfg, "cfg": cfg}, out)
    print(f"[*] Saved model to {out}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/default.yaml")
