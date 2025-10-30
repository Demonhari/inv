# core/embedder.py
import os
from openai import AsyncOpenAI

class Embedder:
    def __init__(self, model="text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        res = await self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in res.data]
