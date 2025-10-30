# agents/evaluator_agent.py
from openai import OpenAI
import json, re
from services.tracing import tracer

SYSTEM = (
    "Evaluate factual consistency of ANSWER against CONTEXT only. "
    "Return JSON: {\"score\": 1-10, \"explanation\": \"...\"}."
)

class EvaluatorAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, answer: str, context: list[str]) -> tuple[dict, dict]:
        context_text = "\n\n".join(context)
        user = f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}\n\nRespond with JSON only."
        with tracer.run(name="EvaluatorAgent.ChatCompletion", run_type="llm", inputs={"answer_len": len(answer)}) as span:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            usage = (resp.usage.model_dump() if hasattr(resp, "usage") and resp.usage else {})
            m = re.search(r"\{.*\}", raw, re.S)
            data = {"score": 0, "explanation": "Could not parse evaluator output."}
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    pass
            span["set_output"]({"evaluation": data, "usage": usage})
            return data, usage
