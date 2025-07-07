#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

class Request(BaseModel):
    abstract_id: str
    abstract: str

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("models/lora_finetuned")
model = AutoModelForSequenceClassification.from_pretrained("models/lora_finetuned")
nlp = spacy.load("en_ner_bc5cdr_md")

@app.post("/analyze")
def analyze(req: Request):
    enc = tokenizer(req.abstract, truncation=True, padding=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits.squeeze()
    probs = torch.softmax(logits, dim=-1).tolist()
    labels = ["Non-Cancer", "Cancer"]
    classification = {labels[i]: probs[i] for i in range(2)}
    pred_label = labels[int(logits.argmax())]
    doc = nlp(req.abstract)
    diseases = list({ent.text for ent in doc.ents if ent.label_ == "DISEASE"})
    return {
        "abstract_id": req.abstract_id,
        "predicted_label": pred_label,
        "confidence_scores": classification,
        "extracted_diseases": diseases
    }
