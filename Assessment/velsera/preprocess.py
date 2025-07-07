#!/usr/bin/env python3
"""
Build a *single* labelled Parquet file from
  data/raw/Cancer/*.txt
  data/raw/Non-Cancer/*.txt
Outputs:
  data/processed/dataset.parquet
  data/processed/train.jsonl
  data/processed/test.jsonl
"""
import pathlib, re, pandas as pd
from sklearn.model_selection import train_test_split

RAW_ROOT = pathlib.Path("data/raw")
OUT_DIR  = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(txt: str) -> str:
    txt = "\n".join(
        ln for ln in txt.splitlines()
        if not re.match(r'^\s*(PMID|Journal|Author|DOI)', ln)
    )
    txt = re.sub(r'\[\d+\]', '[CITATION]', txt)
    txt = re.sub(r'\([A-Za-z]+ et al\.,? \d{4}\)', '(CITATION)', txt)
    return txt.strip()

records = []
for label in ("Cancer", "Non-Cancer"):
    for f in (RAW_ROOT/label).glob("*.txt"):
        records.append({"abstract": clean(f.read_text(encoding="utf-8")), "label": label})

df = pd.DataFrame.from_records(records)
df.to_parquet(OUT_DIR/"dataset.parquet", index=False)

# create a fixed 80/20 split so every other script agrees
train, test = train_test_split(df, test_size=0.20, stratify=df.label, random_state=42)
train.to_json(OUT_DIR/"train.jsonl", orient="records", lines=True)
test.to_json(OUT_DIR/"test.jsonl",  orient="records", lines=True)

print(f"Wrote {len(df):,} abstracts → {OUT_DIR/'dataset.parquet'}")
print(f"Train: {len(train):,}  Test: {len(test):,}")
