# agents/feedback_agent.py
from openai import OpenAI
from services.tracing import tracer

class FeedbackAgent:
    def __init__(self):
        self.client = OpenAI()

    def suggest_improvement(self, query: str, answer: str, eval_data: dict) -> tuple[str, dict]:
        prompt = (
            f"Query: {query}\nAnswer: {answer}\nEvaluation: {eval_data}\n\n"
            "Briefly suggest how to improve the next answer (1-3 bullet points)."
        )
        with tracer.run(name="FeedbackAgent.ChatCompletion", run_type="llm", inputs={"query": query}) as span:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            out = resp.choices[0].message.content.strip()
            usage = (resp.usage.model_dump() if hasattr(resp, "usage") and resp.usage else {})
            span["set_output"]({"feedback_preview": out[:120], "usage": usage})
            return out, usage
