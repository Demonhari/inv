# services/metrics.py
from __future__ import annotations
import threading
from time import perf_counter
from typing import Dict, Optional

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

_registry = CollectorRegistry()

REQ_COUNTER = Counter("rag_requests_total", "Total /ask requests", registry=_registry)
CACHE_HIT_COUNTER = Counter("rag_cache_hits_total", "Cache hits", registry=_registry)

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency per stage",
    ["stage"],
    registry=_registry,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

TOKENS_USED = Counter(
    "rag_tokens_total",
    "LLM tokens used",
    ["agent", "direction"],  # prompt|completion|total
    registry=_registry,
)

TOTAL_COST_USD = Counter("rag_cost_usd_total", "Estimated USD cost", registry=_registry)

INFLIGHT = Gauge("rag_inflight_requests", "In-flight requests", registry=_registry)

class _Timer:
    def __init__(self, stage: str):
        self.stage = stage
        self.t0 = None
    def __enter__(self):
        self.t0 = perf_counter()
        return self
    def __exit__(self, *exc):
        STAGE_LATENCY.labels(self.stage).observe(perf_counter() - self.t0)

class MetricsRegistry:
    """Thread-safe in-memory summary + Prometheus registry."""
    def __init__(self):
        self._lock = threading.Lock()
        self.summary = {
            "requests": 0,
            "cache_hits": 0,
            "tokens": {"guardrail": 0, "generator": 0, "evaluator": 0, "feedback": 0, "total": 0},
            "cost_usd": 0.0,
            "last_scores": [],  # keep a few recent evaluator scores
        }

    def inc_request(self):
        with self._lock:
            self.summary["requests"] += 1
        REQ_COUNTER.inc()

    def inc_cache_hit(self):
        with self._lock:
            self.summary["cache_hits"] += 1
        CACHE_HIT_COUNTER.inc()

    def add_tokens(self, agent: str, usage: Dict):
        # OpenAI usage shape: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
        p = int(usage.get("prompt_tokens", 0))
        c = int(usage.get("completion_tokens", 0))
        t = int(usage.get("total_tokens", p + c))
        with self._lock:
            self.summary["tokens"][agent] = self.summary["tokens"].get(agent, 0) + t
            self.summary["tokens"]["total"] += t
        TOKENS_USED.labels(agent, "prompt").inc(p)
        TOKENS_USED.labels(agent, "completion").inc(c)
        TOKENS_USED.labels(agent, "total").inc(t)

    def add_cost(self, usd: float):
        with self._lock:
            self.summary["cost_usd"] += float(usd)
        TOTAL_COST_USD.inc(float(usd))

    def add_score(self, score: Optional[int]):
        if score is None:
            return
        with self._lock:
            self.summary["last_scores"] = (self.summary["last_scores"] + [int(score)])[-20:]

    def get_summary(self):
        with self._lock:
            return dict(self.summary)

    def timer(self, stage: str):
        return _Timer(stage)

    def prometheus_text(self) -> bytes:
        return generate_latest(_registry)

metrics = MetricsRegistry()
# --- simple cost estimator for OpenAI gpt-3.5 + embed defaults ---
# Adjust if you switch models. Prices as of 2024-era (illustrative).
OPENAI_PRICES = {
    "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},  # $/token
}

def estimate_cost_usd(model: str, usage: dict) -> float:
    p = int(usage.get("prompt_tokens", 0))
    c = int(usage.get("completion_tokens", 0))
    price = OPENAI_PRICES.get(model)
    if not price:
        # fallback to a conservative estimate
        return (p + c) * 0.000002
    return p * price["prompt"] + c * price["completion"]
