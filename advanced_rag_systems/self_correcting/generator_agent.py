from utils import get_openai_client

class GeneratorAgent:
    def __init__(self):
        self.client = get_openai_client()

    def generate_answer(self, query: str, context: str):
        prompt = f"""
        You are an expert assistant. Use the given context to answer the question accurately.

        Context:
        {context}

        Question: {query}
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
