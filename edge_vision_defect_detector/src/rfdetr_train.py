# src/rfdetr_train.py
import os, yaml, argparse
from pathlib import Path

# RF-DETR variants
from rfdetr import (
    RFDETRNano, RFDETRSmall, RFDETRBase,
    RFDETRMedium, RFDETRLarge
)

VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}

def train(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    d = cfg.get("rfdetr", {})
    dataset_dir = cfg.get("dataset_dir", "labelled_data")
    model_dir = cfg.get("model_dir", "data/models")
    out_sub = d.get("output_subdir", f"rfdetr_{d.get('variant','small')}")
    out_dir = os.path.join(model_dir, out_sub)
    os.makedirs(out_dir, exist_ok=True)

    variant = d.get("variant", "small").lower()
    Model = VARIANTS[variant]
    model = Model()

    kwargs = dict(
        dataset_dir=dataset_dir,
        epochs=int(d.get("epochs", 40)),
        batch_size=int(d.get("batch_size", 2)),
        grad_accum_steps=int(d.get("grad_accum_steps", 8)),
        lr=float(d.get("lr", 1e-4)),
        output_dir=out_dir,
        resolution=int(d.get("resolution", 896)),
        device=d.get("device", "cpu"),
        early_stopping=bool(d.get("early_stopping", True)),
        tensorboard=False,  # turn on if you want logs: pip install rfdetr[metrics]
    )

    # resume if provided
    if "resume" in d and d["resume"]:
        kwargs["resume"] = d["resume"]

    print(f"[RF-DETR] Training {variant} → {out_dir}")
    model.train(**kwargs)

    print("\n[RF-DETR] Done. Best weights are usually at:")
    print(f"  {out_dir}/checkpoint_best_total.pth")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", nargs="?", default="configs/rfdetr.yaml")
    args = ap.parse_args()
    train(args.cfg)
