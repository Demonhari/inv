import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

def patch_features(patch_bgr, hog_cfg, lbp_cfg):
    # grayscale
    gray_u8 = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float32)/255.0

    # HOG on float
    H = hog(gray_f,
            pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
            cells_per_block=tuple(hog_cfg["cells_per_block"]),
            orientations=int(hog_cfg["orientations"]),
            block_norm="L2-Hys",
            feature_vector=True)

    # LBP on uint8 (recommended)
    lbp = local_binary_pattern(gray_u8, P=int(lbp_cfg["P"]), R=float(lbp_cfg["R"]),
                               method=lbp_cfg["method"])
    # histogram of LBP
    n_bins = int(lbp_cfg["P"]+2) if lbp_cfg["method"]=="uniform" else int(lbp_cfg["P"])
    (hist, _) = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return np.concatenate([H, hist]).astype(np.float32)
