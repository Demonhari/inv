#!/usr/bin/env python3
"""
Extract top-k evidence sentences supporting the Cancer / Non-Cancer prediction.
Algorithm:  spaCy sentence split → run the same classifier on each sentence →
           select the k with highest P(Cancer).
"""

from pathlib import Path
from typing import List, Dict, Tuple

import spacy
import torch
from torch.nn.functional import softmax
from transformers import PreTrainedModel, PreTrainedTokenizer

# spaCy tiny core is already installed for SciSpaCy
_NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])


def extract_evidence(
    text: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: torch.device,
    k: int = 3,
) -> List[Dict[str, float]]:
    """Return top-k sentences ranked by P(Cancer)."""
    sents = [s.text.strip() for s in _NLP(text).sents if s.text.strip()]
    if not sents:
        return []

    scores: List[Tuple[str, float]] = []
    with torch.no_grad():
        for s in sents:
            toks = tokenizer(
                s, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(device)
            logits = model(**toks).logits.squeeze()
            p_cancer = softmax(logits, dim=-1)[1].item()  # index-1 == "Cancer"
            scores.append((s, p_cancer))

    # sort & keep top-k
    scores.sort(key=lambda t: t[1], reverse=True)
    return [{"sentence": s, "p_cancer": round(p, 4)} for s, p in scores[:k]]
