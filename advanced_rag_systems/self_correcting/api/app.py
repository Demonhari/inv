# api/app.py
from fastapi import FastAPI
from core.pipeline import SelfCorrectingPipeline

app = FastAPI(title="Self-Correcting RAG API")

pipeline = SelfCorrectingPipeline()

@app.get("/")
def root():
    return {"status": "ok", "message": "Self-Correcting RAG API running"}

@app.get("/ask")
def ask(query: str):
    res = pipeline.run(query)
    return res
