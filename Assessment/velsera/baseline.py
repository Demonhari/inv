#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2

def load_data(path):
    df = pd.read_parquet(path)
    return train_test_split(df['abstract'], df['label'], test_size=0.2, random_state=42)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    X_train, X_val, y_train, y_val = load_data("data/processed/dataset.parquet")
    train_enc = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
    val_enc = tokenizer(X_val.tolist(), truncation=True, padding=True, max_length=512)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_enc['input_ids']), torch.tensor(train_enc['attention_mask']), torch.tensor(y_train.values)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_enc['input_ids']), torch.tensor(val_enc['attention_mask']), torch.tensor(y_val.values)
    )

    args = TrainingArguments(
        output_dir="models/baseline",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    trainer.save_model("models/baseline")

if __name__ == "__main__":
    main()
