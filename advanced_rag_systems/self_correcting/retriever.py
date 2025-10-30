# retriever.py
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions

# Prefer a persistent client so your vectors survive restarts
try:
    ChromaClient = chromadb.PersistentClient  # available in newer chromadb
    _client = ChromaClient(path="./.chroma")
except AttributeError:
    _client = chromadb.Client()

class Retriever:
    def __init__(self, collection_name: str = "rag_docs"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")

        # OpenAI embedding function (cheap + good for tests)
        embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )

        # Create or get collection with the embedder attached
        # (so Chroma knows how to embed adds/queries)
        try:
            self.collection = _client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedder,
            )
        except TypeError:
            # Older Chroma may not support get_or_create_collection
            # Fallback: try create, else get.
            try:
                self.collection = _client.create_collection(
                    name=collection_name,
                    embedding_function=embedder,
                )
            except Exception:
                self.collection = _client.get_collection(name=collection_name)

    def add_documents(self, docs: List[str], ids: List[str]) -> None:
        if not docs:
            return
        if len(docs) != len(ids):
            raise ValueError("docs and ids must have the same length.")
        self.collection.add(documents=docs, ids=ids)

    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        if not query.strip():
            return []
        res = self.collection.query(query_texts=[query], n_results=n_results)
        # Chroma returns a list-of-lists for "documents"
        docs = res.get("documents", [[]])[0] if res else []
        return docs or []
