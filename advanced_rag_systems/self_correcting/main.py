# main.py
import argparse
from retriever import Retriever
from guardrail_agent import GuardrailAgent
from generator_agent import GeneratorAgent
from evaluator_agent import EvaluatorAgent
from logger import stage, info, success

def run(query: str):
    retriever = Retriever()
    guardrail = GuardrailAgent()
    generator = GeneratorAgent()
    evaluator = EvaluatorAgent()

    stage("Retrieve")
    docs = retriever.retrieve(query, n_results=5)
    info(f"Retrieved {len(docs)} passages")

    stage("Guardrail")
    filtered = guardrail.filter_relevance(query, docs)
    info(f"Filtered to {len(filtered)} passages")

    stage("Generate")
    answer = generator.generate_answer(query, filtered)
    success("Answer generated")

    stage("Evaluate")
    verdict = evaluator.evaluate_answer(filtered, answer)
    success(f"Score: {verdict.get('score')}  |  {verdict.get('explanation')}")

    stage("Final Answer")
    print(answer)

def main():
    parser = argparse.ArgumentParser(description="Self-correcting RAG pipeline")
    parser.add_argument("query", nargs="*", help="Your question")
    args = parser.parse_args()
    q = " ".join(args.query).strip() or "What is a self-correcting RAG pipeline?"
    run(q)

if __name__ == "__main__":
    main()
