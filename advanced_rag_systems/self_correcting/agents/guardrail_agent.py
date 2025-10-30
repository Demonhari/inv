# agents/guardrail_agent.py
from openai import OpenAI

class GuardrailAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, query: str, docs: list[str]) -> list[str]:
        prompt = f"""
        Filter only the top 3 passages most relevant to this query:

        Query: {query}
        Documents: {docs}

        Return JSON list of relevant passages.
        """
        res = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return eval(res.choices[0].message.content)
