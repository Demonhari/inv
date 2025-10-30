# core/pipeline.py
from services.logger import log_stage, log_json, Timer
from agents.guardrail_agent import GuardrailAgent
from agents.generator_agent import GeneratorAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.feedback_agent import FeedbackAgent
from core.retriever import Retriever
from services.cache import Cache
from services.tracing import tracer
from services.metrics import metrics, estimate_cost_usd

class SelfCorrectingPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.guardrail = GuardrailAgent()
        self.generator = GeneratorAgent()
        self.evaluator = EvaluatorAgent()
        self.feedback = FeedbackAgent()
        self.cache = Cache()

    def run(self, query: str):
        metrics.inc_request()
        with tracer.run(name="SelfCorrectingPipeline.run", run_type="chain", inputs={"query": query}) as root:
            root_id = root["run_id"]

            cached = self.cache.get(query)
            if cached:
                metrics.inc_cache_hit()
                log_stage("Cache Hit", f"Returning cached result for: {query}")
                root["set_output"]({"cached": True, **cached})
                return cached

            # 1) Retrieve
            with tracer.child(root_id, name="Retriever.search", run_type="tool", inputs={"query": query}) as rspan, metrics.timer("retriever"):
                with Timer() as t:
                    docs = self.retriever.search(query)
                rspan["set_output"]({"docs_count": len(docs)})
            log_stage("Retriever", f"Found {len(docs)} docs in {t.elapsed:.2f}s")

            # 2) Guardrail
            with tracer.child(root_id, name="GuardrailAgent.run", run_type="llm", inputs={"docs_count": len(docs)}) as gspan, metrics.timer("guardrail"):
                filtered, usage_g = self.guardrail.run(query, docs)
                # metrics
                metrics.add_tokens("guardrail", usage_g)
                metrics.add_cost(estimate_cost_usd(usage_g.get("model", ""), usage_g))
                gspan["set_output"]({"filtered_count": len(filtered), "usage": usage_g})
            log_stage("Guardrail", f"Filtered to {len(filtered)} docs")

            # 3) Generate
            with tracer.child(root_id, name="GeneratorAgent.run", run_type="llm", inputs={"query": query}) as gen_span, metrics.timer("generator"):
                answer, usage_a = self.generator.run(query, filtered)
                metrics.add_tokens("generator", usage_a)
                metrics.add_cost(estimate_cost_usd(usage_a.get("model", ""), usage_a))
                gen_span["set_output"]({"answer_preview": answer[:120], "usage": usage_a})
            log_stage("Generator", answer)

            # 4) Evaluate
            with tracer.child(root_id, name="EvaluatorAgent.run", run_type="llm", inputs={"answer_len": len(answer)}) as espan, metrics.timer("evaluator"):
                eval_data, usage_e = self.evaluator.run(answer, filtered)
                metrics.add_tokens("evaluator", usage_e)
                metrics.add_cost(estimate_cost_usd(usage_e.get("model", ""), usage_e))
                metrics.add_score(eval_data.get("score"))
                espan["set_output"]({"evaluation": eval_data, "usage": usage_e})
            log_json(eval_data)

            # 5) Feedback
            with tracer.child(root_id, name="FeedbackAgent.suggest_improvement", run_type="llm") as fspan, metrics.timer("feedback"):
                feedback, usage_f = self.feedback.suggest_improvement(query, answer, eval_data)
                metrics.add_tokens("feedback", usage_f)
                metrics.add_cost(estimate_cost_usd(usage_f.get("model", ""), usage_f))
                fspan["set_output"]({"feedback_preview": feedback[:120], "usage": usage_f})
            log_stage("Feedback", feedback)

            result = {
                "query": query,
                "answer": answer,
                "evaluation": eval_data,
                "feedback": feedback,
                "usage": {
                    "guardrail": usage_g,
                    "generator": usage_a,
                    "evaluator": usage_e,
                    "feedback": usage_f,
                },
                "trace_run_id": root_id,
                "metrics_snapshot": metrics.get_summary(),
            }
            self.cache.set(query, result)
            root["set_output"](result)
            return result
