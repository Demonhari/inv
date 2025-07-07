#!/usr/bin/env python3
"""
baseline.py – train a vanilla DistilBERT classifier on the processed corpus.
"""

import inspect
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_NAME   = "distilbert-base-uncased"
NUM_LABELS   = 2
DATA_PATH    = "data/processed/dataset.parquet"
OUT_DIR      = "models/baseline"
BATCH_TRAIN  = 16
BATCH_EVAL   = 32
EPOCHS       = 3
MAX_SEQ_LEN  = 512
RANDOM_SEED  = 42

# --------------------------------------------------------------------------- #
# Dataset wrapper that returns dicts (what Trainer expects)
# --------------------------------------------------------------------------- #
class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # dict of torch.Tensors
        self.labels    = labels      # 1-D numpy array or list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# --------------------------------------------------------------------------- #
def load_data(path: str):
    df = pd.read_parquet(path)

    label_map = {"Non-Cancer": 0, "Cancer": 1}
    df["label_id"] = df["label"].map(label_map).astype(int)

    return train_test_split(
        df["abstract"],
        df["label_id"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

# --------------------------------------------------------------------------- #
def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # -------------------- data -------------------- #
    X_train, X_val, y_train, y_val = load_data(DATA_PATH)

    enc_train = tok(
        X_train.tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    enc_val = tok(
        X_val.tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    train_ds = EncodedDataset(enc_train, y_train.values)
    val_ds   = EncodedDataset(enc_val,   y_val.values)

    # -------------------- training args (robust to version) -------------- #
    kwargs = dict(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        logging_steps=50,
        seed=RANDOM_SEED,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "epoch"

    args = TrainingArguments(**kwargs)

    # -------------------- train -------------------- #
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"✓ Baseline model saved to {OUT_DIR}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
