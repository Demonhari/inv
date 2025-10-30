from utils import get_openai_client
import json

SYSTEM = (
    "You are a Guardrail Agent. Given a user query and retrieved passages, "
    "return the most relevant 1-3 passages. Respond ONLY as compact JSON:\n"
    '{"passages": ["...", "..."]}'
)

class GuardrailAgent:
    def __init__(self):
        self.client = get_openai_client()

    def filter_relevance(self, query: str, docs: list[str]) -> list[str]:
        user = (
            f"Query:\n{query}\n\n"
            f"Retrieved passages (list):\n{json.dumps(docs, ensure_ascii=False)}"
        )
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        try:
            data = json.loads(txt)
            return data.get("passages", [])[:3]
        except Exception:
            # Fallback: if the model returns plain text, just return the first 2 docs
            return docs[:2]
