# core/document_loader.py
import os
from pathlib import Path
import tiktoken

class DocumentLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _chunk_text(self, text, max_tokens=400, overlap=50):
        toks = self.enc.encode(text)
        for i in range(0, len(toks), max_tokens - overlap):
            yield self.enc.decode(toks[i:i+max_tokens])

    def load_all(self):
        docs, ids = [], []
        for path in self.data_dir.glob("*"):
            if path.suffix not in [".txt", ".md"]:
                continue
            text = path.read_text(encoding="utf-8")
            for i, chunk in enumerate(self._chunk_text(text)):
                docs.append(chunk)
                ids.append(f"{path.stem}-{i}")
        return docs, ids
