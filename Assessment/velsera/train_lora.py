#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_MODEL = "distilbert-base-uncased"
NUM_LABELS = 2

def load_data(path):
    df = pd.read_parquet(path)
    return train_test_split(df['abstract'], df['label'], test_size=0.2, random_state=42)

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

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
        output_dir="models/lora_finetuned",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()
    model.save_pretrained("models/lora_finetuned")
    tokenizer.save_pretrained("models/lora_finetuned")

if __name__ == "__main__":
    main()
