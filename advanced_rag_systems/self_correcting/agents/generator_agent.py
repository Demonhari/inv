# agents/generator_agent.py
from openai import OpenAI

class GeneratorAgent:
    def __init__(self):
        self.client = OpenAI()

    def run(self, query: str, context: list[str]) -> str:
        context_text = "\n\n".join(context)
        prompt = f"Use only this context to answer:\n{context_text}\n\nQ: {query}"
        res = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
