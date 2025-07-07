#!/usr/bin/env python3
"""
─────────────────────────────────────────────────────────────────────────────
 FastAPI micro-service for the Velsera research-paper pipeline
─────────────────────────────────────────────────────────────────────────────

POST /predict
Request JSON:
    {"text": "<PubMed abstract …>"}

Response JSON:
    {
      "label": "Cancer",
      "proba": {
        "Non-Cancer": 0.0274,
        "Cancer":     0.9726
      }
    }

If a LoRA-fine-tuned checkpoint is present it is preferred; otherwise the
baseline model is loaded; otherwise we fall back to the raw HF checkpoint
(distilbert-base-uncased) so the API still starts.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.nn.functional import softmax
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ───────────────────────────────────────────────────────────
#  Constants & helper to load the best available checkpoint
# ───────────────────────────────────────────────────────────
MODEL_DIRS = ["models/lora_finetuned", "models/baseline"]
BASE_CHECKPOINT = "distilbert-base-uncased"
LABELS = ["Non-Cancer", "Cancer"]        # idx 0 / idx 1


def _load_checkpoint() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    """Return (tokenizer, model, source_name)."""
    for path in MODEL_DIRS:
        if Path(path).exists():
            return (
                AutoTokenizer.from_pretrained(path),
                AutoModelForSequenceClassification.from_pretrained(path),
                path,
            )
    # Fallback – raw HF weights (un-trained)
    return (
        AutoTokenizer.from_pretrained(BASE_CHECKPOINT),
        AutoModelForSequenceClassification.from_pretrained(
            BASE_CHECKPOINT, num_labels=len(LABELS)
        ),
        BASE_CHECKPOINT,
    )


tokenizer, model, _CKPT_NAME = _load_checkpoint()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ───────────────────────────────────────────────────────────
#  FastAPI plumbing
# ───────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    text: str


app = FastAPI(
    title="Velsera Research-Paper Classifier",
    description="Predict *Cancer* vs *Non-Cancer* and return per-label probabilities.",
    version="0.2.0",
)


@app.get("/")
def health():
    return {"status": "ok", "checkpoint": _CKPT_NAME, "device": str(device)}


@app.post("/predict")
def predict(req: InferenceRequest):
    # Sanity check ----------------------------------------------------------
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Tokenise --------------------------------------------------------------
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    # Infer -----------------------------------------------------------------
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()  # shape: (2,)

    probs = softmax(logits, dim=-1).cpu().tolist()
    pred_idx = int(torch.argmax(logits))

    return {
        "label": LABELS[pred_idx],
        "proba": {LABELS[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ───────────────────────────────────────────────────────────
#  Dev helper:  python app.py  →  auto-reloading server
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
