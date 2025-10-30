# agents/feedback_agent.py
from openai import OpenAI

class FeedbackAgent:
    def __init__(self):
        self.client = OpenAI()

    def suggest_improvement(self, query: str, answer: str, eval_data: dict):
        prompt = f"""
        Query: {query}
        Answer: {answer}
        Evaluation: {eval_data}

        Suggest how the generation step could be improved (briefly).
        """
        res = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
