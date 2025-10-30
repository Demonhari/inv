# agents/evaluator_agent.py
from openai import OpenAI
import json

class EvaluatorAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, answer: str, context: list[str]) -> dict:
        prompt = f"""
        Evaluate factual accuracy of this answer based on context.
        Return JSON like {{"score": 1-10, "explanation": "..."}}

        Answer: {answer}
        Context: {context}
        """
        res = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            return json.loads(res.choices[0].message.content)
        except Exception:
            return {"score": 5, "explanation": "Could not parse output."}
