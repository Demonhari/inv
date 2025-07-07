#!/usr/bin/env python3
import os, re, pandas as pd
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)
def normalize_citations(text):
    text = re.sub(r'\[\d+\]', '[CITATION]', text)
    text = re.sub(r'\([A-Za-z]+ et al\\., \\d{4}\)', '(CITATION)', text)
    return text
def clean_metadata(text):
    lines = text.splitlines()
    return "\\n".join([ln for ln in lines if not re.match(r'^(Journal|DOI|Author):', ln)])
def process_file(fname):
    df = pd.read_csv(os.path.join(RAW_DIR, fname))
    df = df.dropna(subset=['abstract'])
    df['abstract'] = df['abstract'].apply(clean_metadata).apply(normalize_citations)
    df.to_parquet(os.path.join(OUT_DIR, fname.replace('.csv','.parquet')), index=False)
    print(f"Processed {len(df)} records into {OUT_DIR}")
if __name__ == "__main__":
    for f in os.listdir(RAW_DIR):
        if f.endswith(".csv") or f.endswith(".tsv"):
            process_file(f)
