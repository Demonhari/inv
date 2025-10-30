import chromadb
from chromadb.utils import embedding_functions

class Retriever:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("rag_docs")
        self.embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def add_documents(self, docs: list[str], ids: list[str]):
        self.collection.add(documents=docs, ids=ids)

    def retrieve(self, query: str, n_results: int = 3):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0]
