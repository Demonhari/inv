# core/retriever.py
import os
import chromadb
from chromadb.utils import embedding_functions

class Retriever:
    def __init__(self, collection_name="rag_docs"):
        self.client = chromadb.PersistentClient(path="./.chroma")
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn
        )

    def add_documents(self, docs: list[str], ids: list[str]):
        self.collection.add(documents=docs, ids=ids)

    def search(self, query: str, n_results=5) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]
