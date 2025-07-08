#!/usr/bin/env python3
"""
agent.py
────────────────────────────────────────────────────────────────────────────
Glue everything together in one shot:

 1️⃣  Classify an abstract via the local FastAPI
 2️⃣  Extract diseases with SciSpaCy
 3️⃣  Retrieve top-k evidence sentences (FAISS)
 4️⃣  Ask an OpenAI chat model to summarise in ≤ 3 bullet points
"""

from __future__ import annotations
import os, sys, json, requests
from typing import Any, Dict, List

import openai                                   # ← no more langchain-openai
from extract_diseases import extract_diseases_spacy
from confidence_retrieval import get_evidence   # FAISS-backed helper

# ────────────────────────────
#  Config
# ────────────────────────────
API_URL        = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")           # must be set
TOP_K          = int(os.getenv("TOP_K_EVIDENCE", "3"))

if not OPENAI_API_KEY:
    raise SystemExit("❌  set OPENAI_API_KEY env-var before running the agent")
openai.api_key = OPENAI_API_KEY


# ────────────────────────────
#  Helpers
# ────────────────────────────
def _classify(text: str) -> Dict[str, Any]:
    r = requests.post(API_URL, json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.json()


def _format_evidence(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No external evidence found."
    return "\n".join(f"- {r['sentence']} (PubMed ID {r['pmid']})" for r in rows)


def _ask_llm(prompt: str) -> str:
    """Single call to the ChatCompletion endpoint."""
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ────────────────────────────
#  Main agent
# ────────────────────────────
def run_agent(text: str, k: int = TOP_K) -> Dict[str, Any]:
    cls      = _classify(text)
    diseases = extract_diseases_spacy(text)
    evidence = get_evidence(text, k=k)

    # Early exit for non-cancer abstracts
    if cls["label"] == "Non-Cancer":
        return {
            "classification": cls,
            "diseases": diseases,
            "evidence": evidence,
            "summary": "The abstract is predicted *Non-Cancer* – no summary generated.",
        }

    # Build one compact prompt
    prompt = (
        "You are a biomedical research assistant.\n"
        "Respond in **three concise bullet points** citing diseases and evidence.\n\n"
        f"Abstract:\n{text}\n\n"
        f"Predicted label: Cancer (confidence {cls['confidence']:.2%}).\n"
        f"Diseases extracted: {', '.join(diseases) or 'none'}.\n"
        f"External evidence:\n{_format_evidence(evidence)}\n\n"
        "Draft the summary now:"
    )

    summary = _ask_llm(prompt)

    return {
        "classification": cls,
        "diseases": diseases,
        "evidence": evidence,
        "summary": summary,
    }


# ────────────────────────────
#  CLI usage
# ────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py 'BRCA1 mutation raises breast-cancer risk'")
        sys.exit(1)

    result = run_agent(" ".join(sys.argv[1:]))
    print(json.dumps(result, indent=2))
