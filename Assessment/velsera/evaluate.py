#!/usr/bin/env python3
import json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_preds(model_dir, data_file):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)

    texts, labels = [], []
    with open(data_file) as f:
        for ln in f:
            obj = json.loads(ln)
            texts.append(obj["abstract"])
            labels.append(obj["label"])

    # --- map string → int so it matches model output ---
    label_map = {"Non-Cancer": 0, "Cancer": 1}
    y_true = [label_map[x] for x in labels]

    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    y_pred = torch.argmax(logits, dim=-1).tolist()

    return y_true, y_pred


if __name__ == "__main__":
    for name, path in [("Baseline","models/baseline"), ("LoRA Fine-Tuned","models/lora_finetuned")]:
        y_true, y_pred = load_preds(path, "data/processed/test.jsonl")
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.2%}")
        print(f"F1-score: {f1:.2f}")
        print("Confusion Matrix:")
        print("        Pred Cancer  Pred Non-Cancer")
        print(f"Actual C     {cm[1,1]:4d}           {cm[1,0]:4d}")
        print(f"Actual NC    {cm[0,1]:4d}           {cm[0,0]:4d}")
