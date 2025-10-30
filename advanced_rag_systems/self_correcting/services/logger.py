# services/logger.py
from rich.console import Console
from datetime import datetime
import json
import time

console = Console()

def log_stage(stage: str, msg: str):
    console.rule(f"[bold cyan]{stage}")
    console.print(msg)

def log_json(data: dict):
    console.print_json(json.dumps(data, indent=2))

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
