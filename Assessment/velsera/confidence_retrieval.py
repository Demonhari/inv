#!/usr/bin/env python3
"""
confidence_retrieval.py
────────────────────────────────────────────────────────────────────────────
1.  Calls /predict (local FastAPI) or the model in-process.
2.  Uses the predicted label **probability** as a filter
    → only search PubMed if confidence < thresh.
3.  Fetches top-k abstracts via Entrez E-utils,
   embeds & ranks them with LangChain,
   then returns title + link + similarity score.

Run:
    python confidence_retrieval.py "text of abstract..."
"""
from __future__ import annotations

import os, requests, json
from typing import List, Dict
from langchain_community.document_loaders import PubMedLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

PREDICT_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
THRESH      = 0.90           # only trigger retrieval if < 90 % confident
TOP_K       = 3

def classify(text: str) -> Dict:
    r = requests.post(PREDICT_URL, json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_pubmed(query: str, k: int = 10) -> List[str]:
    loader = PubMedLoader(query, max_articles=k)
    docs   = loader.load()
    return [d.page_content for d in docs]

def rank_documents(query: str, docs: List[str], k: int = 3):
    splitter   = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1024)
    splits     = splitter.create_documents(docs)
    db         = FAISS.from_documents(splits, OpenAIEmbeddings())
    results    = db.similarity_search_with_score(query, k=k)
    return [
        {"snippet": doc.page_content[:250] + "…", "score": round(score, 3)}
        for doc, score in results
    ]

def explain(text: str):
    pred = classify(text)
    conf = pred["confidence"]
    if conf >= THRESH:
        pred["evidence"] = []
        return pred

    # low-confidence → search & rank
    docs      = fetch_pubmed(text, k=20)
    evidence  = rank_documents(text, docs, k=TOP_K)
    pred["evidence"] = evidence
    return pred

if __name__ == "__main__":
    import sys, pprint
    pprint.pp(explain(sys.argv[1]))
