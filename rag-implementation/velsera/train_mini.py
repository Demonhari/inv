#!/usr/bin/env python3
"""
Fine-tunes microsoft/MiniLM-L6-v2 with the same LoRA config as DistilBERT.
Outputs to  models/minilm_finetuned/
Takes ≈ 3-4 min on CPU (22 M base params, 0.4 M trainable via LoRA).
"""

from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import pandas as pd

BASE_MODEL = "microsoft/MiniLM-L6-v2"
NUM_LABELS = 2
OUT_DIR = Path("models/minilm_finetuned")


def load_splits():
    df = pd.read_parquet("data/processed/dataset.parquet")
    return df["abstract"].tolist(), df["label_id"].tolist()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS
    )
    peft_cfg = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    X, y = load_splits()
    enc = tokenizer(
        X, truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    ds = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y))

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=16,
        num_train_epochs=5,
        learning_rate=5e-4,
        logging_steps=25,
        save_strategy="no",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
