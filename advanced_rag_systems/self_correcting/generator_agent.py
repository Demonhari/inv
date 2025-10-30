from utils import get_openai_client

SYSTEM = (
    "You are a careful, grounded QA model. Use ONLY the provided context. "
    "If the answer isn't in the context, say you don't have enough info."
)

class GeneratorAgent:
    def __init__(self):
        self.client = get_openai_client()

    def generate_answer(self, query: str, context: list[str] | str):
        if isinstance(context, list):
            context = "\n\n".join(context)

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer concisely and cite snippets by quoting short phrases from the context."
        )
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
