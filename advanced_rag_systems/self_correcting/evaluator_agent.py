from utils import get_openai_client

class EvaluatorAgent:
    def __init__(self):
        self.client = get_openai_client()

    def evaluate_answer(self, context: str, answer: str):
        prompt = f"""
        You are an Evaluator Agent. Compare the answer to the provided context and score its factual consistency from 1 to 10.
        Provide a short justification.

        Context: {context}
        Answer: {answer}

        Respond in JSON:
        {{
            "score": <integer>,
            "explanation": "<text>"
        }}
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
