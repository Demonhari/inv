from utils import get_openai_client

class GuardrailAgent:
    def __init__(self):
        self.client = get_openai_client()

    def filter_relevance(self, query: str, docs: list[str]):
        prompt = f"""
        You are a Guardrail Agent. Given the query and retrieved documents,
        rank and filter only the most relevant ones.

        Query: {query}

        Documents:
        {docs}

        Return only the most relevant documents as a list.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
