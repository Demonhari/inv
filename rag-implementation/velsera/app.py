#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.nn.functional import softmax
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evidence import extract_evidence   # ← NEW

# ───────────────────────────────────────────────────────────
#  Constants & helper to load the best available checkpoint
# ───────────────────────────────────────────────────────────
MODEL_DIRS = [
    "models/lora_finetuned",      # DistilBERT-LoRA
    "models/minilm_finetuned",    # MiniLM-LoRA  ← new
    "models/baseline",            # full fine-tune
]
BASE_CHECKPOINT = "distilbert-base-uncased"
LABELS = ["Non-Cancer", "Cancer"]        # idx 0 / idx 1


def _load_checkpoint() -> Tuple:
    for p in MODEL_DIRS:
        if Path(p).exists():
            return (
                AutoTokenizer.from_pretrained(p),
                AutoModelForSequenceClassification.from_pretrained(p),
                p,
            )
    # fallback (un-trained)
    return (
        AutoTokenizer.from_pretrained(BASE_CHECKPOINT),
        AutoModelForSequenceClassification.from_pretrained(
            BASE_CHECKPOINT, num_labels=len(LABELS)
        ),
        BASE_CHECKPOINT,
    )


tokenizer, model, _CKPT = _load_checkpoint()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ───────────────────────────────────────────────────────────
#  FastAPI
# ───────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    text: str = Field(..., description="PubMed abstract or free text")
    k: int = Field(
        0,
        ge=0,
        le=10,
        description="Return the top-k evidence sentences (0 = disabled).",
    )


app = FastAPI(
    title="Velsera Research-Paper Classifier",
    description="Predict Cancer vs Non-Cancer and optionally return evidence.",
    version="0.3.0",
)


@app.get("/")
def health():
    return {"status": "ok", "checkpoint": _CKPT, "device": str(device)}


@app.post("/predict")
def predict(req: InferenceRequest):
    if not req.text.strip():
        raise HTTPException(400, "Input text cannot be empty.")

    toks = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**toks).logits.squeeze()

    probs = softmax(logits, dim=-1).tolist()
    pred_idx = int(torch.argmax(logits))
    response: Dict = {
        "label": LABELS[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {LABELS[i]: round(p, 4) for i, p in enumerate(probs)},
    }

    if req.k:
        response["evidence"] = extract_evidence(
            req.text, tokenizer, model, device, k=req.k
        )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
