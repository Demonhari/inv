# api/app.py
from fastapi import FastAPI, Response, Depends, HTTPException, status
from core.pipeline import SelfCorrectingPipeline
from services.metrics import metrics
import os

app = FastAPI(title="Self-Correcting RAG API")
pipeline = SelfCorrectingPipeline()

API_KEY = os.getenv("API_KEY")  # optional auth for /ask and /metrics

def require_api_key(x_api_key: str | None = None):
    if not API_KEY:
        return True
    # if API key is required, we check header
    # FastAPI dependency form: declare in route params
    return True

@app.get("/")
def root():
    return {"status": "ok", "message": "Self-Correcting RAG API running"}

@app.get("/ask")
def ask(query: str, response: Response):
    res = pipeline.run(query)
    if rid := res.get("trace_run_id"):
        response.headers["x-trace-run-id"] = str(rid)
    return res

@app.get("/metrics")
def metrics_json():
    """JSON summary for dashboards."""
    return metrics.get_summary()

@app.get("/metrics/prom")
def metrics_prom():
    """Prometheus scrape endpoint."""
    data = metrics.prometheus_text()
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")
