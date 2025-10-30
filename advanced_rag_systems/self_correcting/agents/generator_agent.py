# agents/generator_agent.py
from openai import OpenAI
from services.tracing import tracer

SYSTEM = (
    "You are a careful, grounded QA model. Use ONLY the provided context. "
    "If the answer isn't in the context, say so."
)

class GeneratorAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, query: str, context: list[str]) -> tuple[str, dict]:
        context_text = "\n\n".join(context)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer concisely."
        with tracer.run(name="GeneratorAgent.ChatCompletion", run_type="llm", inputs={"query": query}) as span:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            out = resp.choices[0].message.content.strip()
            usage = (resp.usage.model_dump() if hasattr(resp, "usage") and resp.usage else {})
            span["set_output"]({"answer_preview": out[:120], "usage": usage})
            return out, usage
