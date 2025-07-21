#!/usr/bin/env python3
"""
Evaluate *all* checkpoints in ./models/* on the held-out test split.

Outputs a nicely formatted report to stdout *and* (optionally) writes the
metrics to `output/metrics.json` so the CI workflow or a notebook can pick
them up later.

Assumes:
    • data/processed/test.jsonl   (built by preprocess.py)
    • every valid model dir has a Hugging-Face   config.json
"""

from __future__ import annotations
from pathlib import Path
import json
import sys
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEST_FILE  = Path("data/processed/test.jsonl")
MODELS_DIR = Path("models")
OUT_JSON   = Path("output/metrics.json")        # written only if parent exists
LABELS     = ["Non-Cancer", "Cancer"]
LABEL_MAP  = {lbl: idx for idx, lbl in enumerate(LABELS)}   # str → int


# ───────────────────────── helpers ──────────────────────────
def list_models() -> list[Path]:
    """Return every sub-dir that *contains* a config.json."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p for p in MODELS_DIR.iterdir() if (p / "config.json").exists())


def load_test() -> tuple[list[str], list[int]]:
    """Return texts & int-labels from the test split."""
    texts, y = [], []
    with TEST_FILE.open() as fh:
        for line in fh:
            obj = json.loads(line)
            texts.append(obj["abstract"])
            y.append(LABEL_MAP[obj["label"]])
    return texts, y


def predict(model_dir: Path, texts: list[str]) -> list[int]:
    tok  = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    mdl  = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
    mdl.eval()
    enc  = tok(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**enc).logits
    return torch.argmax(logits, dim=-1).tolist()


# ───────────────────────── main ──────────────────────────
def main() -> None:
    if not TEST_FILE.exists():
        sys.exit("❌  run `python preprocess.py` first – test split missing.")

    texts, y_true = load_test()
    all_metrics: dict[str, dict] = {}

    for model_dir in list_models():
        pretty = model_dir.name.replace("_", " ").title()      # “lora_finetuned” → “Lora Finetuned”
        y_pred = predict(model_dir, texts)

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        prec, rec, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        # ── console report ─────────────────────────────────
        print(f"\n=== {pretty}  ({model_dir}) ===")
        print(f"Accuracy : {acc:.2%}")
        print(f"Macro-F1 : {f1:.2f}")
        print("Label metrics:")
        for i, lbl in enumerate(LABELS):
            print(f"  {lbl:<11}  P={prec[i]:.2f}  R={rec[i]:.2f}  F1={f1s[i]:.2f}")
        print("Confusion-matrix")
        print("             Pred Cancer  Pred Non-Cancer")
        print(f"Actual C         {cm[1,1]:4d}             {cm[1,0]:4d}")
        print(f"Actual NC        {cm[0,1]:4d}             {cm[0,0]:4d}")

        # ── collect for JSON ───────────────────────────────
        all_metrics[pretty] = {
            "accuracy": acc,
            "macro_f1": f1,
            "per_label": {
                LABELS[i]: {"precision": prec[i], "recall": rec[i], "f1": f1s[i]}
                for i in range(len(LABELS))
            },
            "confusion": cm.tolist(),
        }

    # optional write-out
    if OUT_JSON.parent.exists():
        OUT_JSON.write_text(json.dumps(all_metrics, indent=2))
        print(f"\n📄  Metrics written to {OUT_JSON}")


if __name__ == "__main__":
    main()
