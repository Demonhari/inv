#!/usr/bin/env python3
"""
Build a *single* labelled Parquet file from

    data/raw/Cancer/*.txt
    data/raw/Non-Cancer/*.txt

Outputs (default):
    data/processed/dataset.parquet
    data/processed/train.jsonl
    data/processed/test.jsonl

Override locations with env-vars, e.g.:

    RAW_ROOT=/my/raw/dir   OUT_DIR=/tmp/out   python preprocess.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
#  I/O locations (can be overridden during tests / CI)
# ──────────────────────────────────────────────────────────────
RAW_ROOT: Path = Path(os.getenv("RAW_ROOT", "data/raw")).resolve()
OUT_DIR:  Path = Path(os.getenv("OUT_DIR",  "data/processed")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def _clean(txt: str) -> str:
    """Remove PubMed headers + normalise citations."""
    txt = "\n".join(
        ln for ln in txt.splitlines()
        if not re.match(r"^\s*(PMID|Journal|Author|DOI)", ln)
    )
    txt = re.sub(r"\[\d+\]", "[CITATION]", txt)                       # numeric refs
    txt = re.sub(r"\([A-Za-z]+ et al\.,? \d{4}\)", "(CITATION)", txt) # (Smith et al., 2021)
    return txt.strip()


def _iter_raw():
    """Yield {"abstract": str, "label": str} for every .txt file."""
    for label in ("Cancer", "Non-Cancer"):
        for fp in (RAW_ROOT / label).glob("*.txt"):
            yield {"abstract": _clean(fp.read_text(encoding="utf-8")), "label": label}


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    if not RAW_ROOT.exists():
        raise SystemExit(f"❌  RAW_ROOT not found: {RAW_ROOT}")

    df = pd.DataFrame(_iter_raw())
    if df.empty:
        raise SystemExit("❌  No .txt files found – did you unzip the dataset?")

    # full parquet
    parquet_path = OUT_DIR / "dataset.parquet"
    df.to_parquet(parquet_path, index=False)

    # stratified 80/20 split – deterministic
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df.label, random_state=42
    )
    train_df.to_json(OUT_DIR / "train.jsonl", orient="records", lines=True)
    test_df.to_json(OUT_DIR / "test.jsonl",  orient="records", lines=True)

    # ─── logging (robust when OUT_DIR is outside repo) ──────────────────────
    try:
        pretty = parquet_path.relative_to(Path.cwd())
    except ValueError:
        pretty = parquet_path
    print(f"✓ Wrote {len(df):,} abstracts → {pretty}")
    print(f"   Train: {len(train_df):,}   Test: {len(test_df):,}")
    print(f"   OUT_DIR = {OUT_DIR}")


if __name__ == "__main__":
    main()
