from retriever import Retriever
from guardrail_agent import GuardrailAgent
from generator_agent import GeneratorAgent
from evaluator_agent import EvaluatorAgent

def main():
    # Step 1: Initialize modules
    retriever = Retriever()
    guardrail = GuardrailAgent()
    generator = GeneratorAgent()
    evaluator = EvaluatorAgent()

    # Step 2: Add sample documents (replace with your corpus)
    docs = [
        "RAG stands for Retrieval Augmented Generation, combining search with LLMs.",
        "The self-correcting RAG pipeline uses multiple LLMs for retrieval, filtering, generation, and evaluation.",
        "OpenAI GPT models are often used in RAG systems for context-based question answering."
    ]
    retriever.add_documents(docs, ids=["1", "2", "3"])

    # Step 3: User query
    query = "What is a self-correcting RAG pipeline?"

    # Step 4: Retrieve documents
    retrieved = retriever.retrieve(query)
    print("\nRetrieved Docs:", retrieved)

    # Step 5: Guardrail filtering
    filtered_docs = guardrail.filter_relevance(query, retrieved)
    print("\nFiltered Context:", filtered_docs)

    # Step 6: Generate answer
    answer = generator.generate_answer(query, filtered_docs)
    print("\nGenerated Answer:", answer)

    # Step 7: Evaluate answer
    evaluation = evaluator.evaluate_answer(filtered_docs, answer)
    print("\nEvaluation:", evaluation)


if __name__ == "__main__":
    main()
