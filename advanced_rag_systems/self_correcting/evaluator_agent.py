from utils import get_openai_client
import json
import re

SYSTEM = (
    "You are an Evaluator Agent. Compare the ANSWER against CONTEXT only. "
    "Score factual consistency from 1-10 and justify briefly. Output JSON only."
)

class EvaluatorAgent:
    def __init__(self):
        self.client = get_openai_client()

    def evaluate_answer(self, context: list[str] | str, answer: str):
        if isinstance(context, list):
            context = "\n\n".join(context)

        user = (
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            'Respond as JSON like {"score": 7, "explanation": "..."}'
        )
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # be tolerant to stray text
        m = re.search(r"\{.*\}", raw, re.S)
        data = {"score": 0, "explanation": "Could not parse evaluator output."}
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass
        return data
