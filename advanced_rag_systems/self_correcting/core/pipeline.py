# core/pipeline.py
from services.logger import log_stage, log_json, Timer
from agents.guardrail_agent import GuardrailAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.feedback_agent import FeedbackAgent
from core.retriever import Retriever
from services.cache import Cache

class SelfCorrectingPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.guardrail = GuardrailAgent()
        self.generator = GeneratorAgent()
        self.evaluator = EvaluatorAgent()
        self.feedback = FeedbackAgent()
        self.cache = Cache()

    def run(self, query: str):
        # Cache hit check
        cached = self.cache.get(query)
        if cached:
            log_stage("Cache Hit", f"Returning cached result for: {query}")
            return cached

        # 1. Retrieve
        with Timer() as t:
            docs = self.retriever.search(query)
        log_stage("Retriever", f"Found {len(docs)} docs in {t.elapsed:.2f}s")

        # 2. Guardrail
        filtered = self.guardrail.run(query, docs)
        log_stage("Guardrail", f"Filtered to {len(filtered)} docs")

        # 3. Generate
        answer = self.generator.run(query, filtered)
        log_stage("Generator", answer)

        # 4. Evaluate
        eval_data = self.evaluator.run(answer, filtered)
        log_json(eval_data)

        # 5. Feedback
        feedback = self.feedback.suggest_improvement(query, answer, eval_data)
        log_stage("Feedback", feedback)

        result = {
            "query": query,
            "answer": answer,
            "evaluation": eval_data,
            "feedback": feedback,
        }

        self.cache.set(query, result)
        return result
