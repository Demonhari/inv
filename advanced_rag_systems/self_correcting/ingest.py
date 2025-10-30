# ingest.py
import os
from pathlib import Path
from typing import Iterable, List

import tiktoken
from retriever import Retriever
from logger import stage, info, success, warn

ENC = tiktoken.get_encoding("cl100k_base")

def chunk(text: str, max_tokens: int = 350, overlap: int = 40) -> List[str]:
    toks = ENC.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        window = toks[i : i + max_tokens]
        chunks.append(ENC.decode(window))
        i += max_tokens - overlap
    return chunks

def read_files(paths: Iterable[Path]) -> list[tuple[str, str]]:
    docs = []
    for p in paths:
        if p.suffix.lower() not in {".txt", ".md"}:
            warn(f"Skipping unsupported file: {p.name}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for j, ch in enumerate(chunk(text)):
            doc_id = f"{p.stem}-{j}"
            docs.append((doc_id, ch))
    return docs

def main(data_dir: str = "data"):
    stage("Ingestion", f"Scanning folder: {data_dir}")
    folder = Path(data_dir)
    if not folder.exists():
        warn("No data/ folder found. Create one and add .txt/.md files.")
        return

    files = list(folder.rglob("*"))
    pairs = read_files(files)

    if not pairs:
        warn("No ingestible files found.")
        return

    retriever = Retriever()
    ids = [pid for pid, _ in pairs]
    docs = [txt for _, txt in pairs]
    retriever.add_documents(docs, ids)
    success(f"Ingested {len(docs)} chunks from {len(files)} files.")

if __name__ == "__main__":
    main()
