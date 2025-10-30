from fastapi import FastAPI, Response
from core.pipeline import SelfCorrectingPipeline

app = FastAPI(title="Self-Correcting RAG API")
pipeline = SelfCorrectingPipeline()

@app.get("/ask")
def ask(query: str, response: Response):
    res = pipeline.run(query)
    if rid := res.get("trace_run_id"):
        response.headers["x-trace-run-id"] = str(rid)
    return res
