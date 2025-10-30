# agents/guardrail_agent.py
from openai import OpenAI
import json
from services.tracing import tracer

class GuardrailAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, query: str, docs: list[str]) -> tuple[list[str], dict]:
        prompt = (
            "Return a JSON array of the top 3 passages most relevant to the query.\n"
            f"Query: {query}\nDocuments: {json.dumps(docs, ensure_ascii=False)}"
        )
        with tracer.run(name="GuardrailAgent.ChatCompletion", run_type="llm", inputs={"prompt_preview": prompt[:200]}) as span:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = resp.choices[0].message.content.strip()
            usage = (resp.usage.model_dump() if hasattr(resp, "usage") and resp.usage else {})
            try:
                passages = json.loads(txt)
                if isinstance(passages, list):
                    passages = passages[:3]
                else:
                    passages = docs[:3]
            except Exception:
                passages = docs[:3]
            span["set_output"]({"passages_count": len(passages), "usage": usage})
            return passages, usage
