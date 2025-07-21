#!/usr/bin/env python3
"""
train_lora.py – LoRA-fine-tune DistilBERT on the processed corpus.
Outputs:
    models/lora_finetuned/  (adapter weights + tokenizer)
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
from peft import get_peft_model, LoraConfig

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_MODEL  = "distilbert-base-uncased"
NUM_LABELS  = 2
DATA_PATH   = "data/processed/dataset.parquet"
OUT_DIR     = "models/lora_finetuned"
BATCH_TRAIN = 8
BATCH_EVAL  = 16
EPOCHS      = 5
MAX_SEQ_LEN = 512
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Dataset wrapper – returns dicts
# --------------------------------------------------------------------------- #
class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# --------------------------------------------------------------------------- #
def load_data(path: str):
    df = pd.read_parquet(path)
    df["label_id"] = df["label"].map({"Non-Cancer": 0, "Cancer": 1}).astype(int)

    return train_test_split(
        df["abstract"],
        df["label_id"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label_id"],
    )

# --------------------------------------------------------------------------- #
def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )

    # ----- LoRA config ------------------------------------------------------ #
    peft_cfg = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_lin", "k_lin", "v_lin"],
    )
    model = get_peft_model(base_model, peft_cfg)
    model.print_trainable_parameters()

    # ----- data ------------------------------------------------------------- #
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

    # ----- TrainingArguments (robust) -------------------------------------- #
    kwargs = dict(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=3e-4,
        logging_steps=50,
        seed=RANDOM_SEED,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "epoch"

    args = TrainingArguments(**kwargs)

    # ----- train ------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"✓ LoRA-finetuned model saved to {OUT_DIR}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
