# services/metrics.py
import time

class Metrics:
    def __init__(self):
        self.records = []

    def record(self, stage, tokens, cost_usd, duration):
        self.records.append({
            "stage": stage,
            "tokens": tokens,
            "cost_usd": cost_usd,
            "duration_s": round(duration, 2)
        })

    def summary(self):
        total_cost = sum(r["cost_usd"] for r in self.records)
        total_time = sum(r["duration_s"] for r in self.records)
        return {"total_cost_usd": total_cost, "total_time_s": total_time}
