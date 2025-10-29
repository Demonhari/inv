import os, glob, yaml
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
from .models_cnn import build_model

class PatchDS(Dataset):
    """
    Loads positive and negative patch PNGs from disk.
    `neg_dir` can be a single directory (str) or a list/tuple of directories.
    """
    def __init__(self, pos_dir, neg_dir, size=32, jitter_b=0.0, jitter_c=0.0, train=True):
        self.items = []
        # positives
        for p in glob.glob(os.path.join(pos_dir, "*.png")):
            self.items.append((p, 1))
        # negatives
        neg_dirs = neg_dir if isinstance(neg_dir, (list, tuple)) else [neg_dir]
        for nd in neg_dirs:
            if nd and os.path.isdir(nd):
                for pth in glob.glob(os.path.join(nd, "*.png")):
                    self.items.append((pth, 0))

        self.size = int(size)
        self.train = bool(train)

        ops = []
        if self.train and (jitter_b > 0 or jitter_c > 0):
            ops.append(transforms.ColorJitter(brightness=jitter_b, contrast=jitter_c))
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            im = np.zeros((self.size, self.size), np.uint8)
        im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # For ColorJitter we briefly go to 3-channels PIL, then back to gray
        im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        pil = transforms.ToPILImage()(im_bgr)
        if self.train:
            pil = self.tf(pil)
        im_aug = np.array(pil)[:, :, 0]  # back to single channel

        ten = torch.from_numpy(im_aug).float().unsqueeze(0) / 255.0  # 1xHxW
        return ten, torch.tensor(float(y), dtype=torch.float32)

def main(cfg_path, arch):
    cfg = yaml.safe_load(open(cfg_path))
    cnn = cfg["cnn"]
    size = int(cfg["patch_size"])

    # patch directories
    train_pos = os.path.join(cfg["patch_root"], "train_pos")
    train_neg = os.path.join(cfg["patch_root"], "train_neg")
    val_pos   = os.path.join(cfg["patch_root"], "val_pos")
    val_neg   = os.path.join(cfg["patch_root"], "val_neg")

    # optional hard negatives
    extra_neg = os.path.join(cfg["patch_root"], "train_neg_hard")
    if not os.path.isdir(extra_neg):
        extra_neg = None

    ds_tr = PatchDS(
        train_pos,
        [train_neg, extra_neg] if extra_neg else [train_neg],
        size,
        cnn.get("jitter_brightness", 0.0),
        cnn.get("jitter_contrast",   0.0),
        train=True
    )
    ds_va = PatchDS(val_pos, val_neg, size, 0.0, 0.0, train=False)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cnn["batch_size"]),
        shuffle=True,
        num_workers=int(cnn.get("num_workers", 0)),
        pin_memory=False
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=int(cnn["batch_size"]),
        shuffle=False,
        num_workers=int(cnn.get("num_workers", 0)),
        pin_memory=False
    )

    device = torch.device("cpu")
    net = build_model(arch).to(device)

    # Loss: BCE or Focal
    pos_w = torch.tensor(float(cnn.get("pos_weight", 1.0)), device=device)
    use_focal = bool(cnn.get("use_focal", False))
    gamma = float(cnn.get("focal_gamma", 2.0))
    alpha = float(cnn.get("focal_alpha", 0.25))

    if use_focal:
        def focal_bce_with_logits(logits, targets):
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            prob = torch.sigmoid(logits)
            p_t = prob * targets + (1 - prob) * (1 - targets)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * (1 - p_t).pow(gamma) * bce
            return loss.mean()
        def loss_fn(logits, y): return focal_bce_with_logits(logits, y)
    else:
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        def loss_fn(logits, y): return bce(logits, y)

    opt = torch.optim.AdamW(net.parameters(), lr=float(cnn["lr"]), weight_decay=float(cnn["weight_decay"]))
    best_loss = float('inf')
    patience = int(cnn.get("patience", 5))
    bad = 0

    for epoch in range(1, int(cnn["epochs"]) + 1):
        net.train()
        tr_loss = 0.0
        for x, y in dl_tr:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= max(1, len(ds_tr))

        net.eval()
        va_loss = 0.0
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device); y = y.to(device)
                logits = net(x)
                loss = loss_fn(logits, y)
                va_loss += loss.item() * x.size(0)
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                tp += ((pred == 1) & (y == 1)).sum().item()
                tn += ((pred == 0) & (y == 0)).sum().item()
                fp += ((pred == 1) & (y == 0)).sum().item()
                fn += ((pred == 0) & (y == 1)).sum().item()
        va_loss /= max(1, len(ds_va))
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        print(f"[{epoch:02d}] train={tr_loss:.4f} val={va_loss:.4f} P={prec:.3f} R={rec:.3f}")

        # save best
        if va_loss < best_loss - 1e-5:
            best_loss = va_loss
            bad = 0
            os.makedirs(cfg["model_dir"], exist_ok=True)
            torch.save({"arch": arch, "state": net.state_dict(), "cfg": cfg},
                       os.path.join(cfg["model_dir"], f"cnn_{arch}.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    print(f"Saved best to {os.path.join(cfg['model_dir'], f'cnn_{arch}.pt')}")

if __name__ == "__main__":
    import sys
    arch = "large"
    if len(sys.argv) >= 3 and sys.argv[1] in ["--arch", "-a"]:
        arch = sys.argv[2]
        cfg = sys.argv[3] if len(sys.argv) >= 4 else "configs/cnn.yaml"
    else:
        cfg = sys.argv[1] if len(sys.argv) >= 2 else "configs/cnn.yaml"
    main(cfg, arch)
