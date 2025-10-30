# services/tracing.py
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    from langsmith import Client  # pip install langsmith
    _HAS_LS = True
except Exception:
    Client = None
    _HAS_LS = False


class Tracer:
    """
    LangSmith-backed tracer with a graceful no-op fallback.
    Set:
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=lsv2_...
      - LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  (optional; defaults)
      - LANGCHAIN_PROJECT=self-correcting-rag (optional)
    """

    def __init__(self, project: Optional[str] = None):
        self.enabled = (
            _HAS_LS
            and os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes"}
            and os.getenv("LANGCHAIN_API_KEY")
        )
        self.project = project or os.getenv("LANGCHAIN_PROJECT", "self-correcting-rag")
        self._client = Client() if self.enabled else None

    @contextmanager
    def run(self, name: str, inputs: Optional[Dict[str, Any]] = None, run_type: str = "chain", parent_id: Optional[str] = None):
        start = time.time()
        run_id = None
        if self.enabled and self._client:
            try:
                run = self._client.create_run(
                    name=name,
                    run_type=run_type,
                    inputs=inputs or {},
                    project_name=self.project,
                    parent_run_id=parent_id,
                    start_time=int(start * 1000),
                )
                run_id = run.id
            except Exception:
                run_id = None  # fall back to no-op if anything goes wrong
        error = None
        outputs: Dict[str, Any] = {}
        try:
            yield {"run_id": run_id, "set_output": lambda d: outputs.update(d if isinstance(d, dict) else {"output": d})}
        except Exception as e:
            error = str(e)
            raise
        finally:
            end = time.time()
            if self.enabled and self._client and run_id:
                try:
                    self._client.update_run(
                        run_id,
                        outputs=outputs,
                        error=error,
                        end_time=int(end * 1000),
                    )
                except Exception:
                    pass

    def child(self, parent_run_id: Optional[str], name: str, **kwargs):
        # convenience wrapper to create a child run
        return self.run(name=name, parent_id=parent_run_id, **kwargs)


# module-level singleton
tracer = Tracer()
