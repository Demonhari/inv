# Self-Correcting RAG (OpenAI + Chroma)

A modular RAG pipeline with **Retriever → Guardrail → Generator → Evaluator** agents.
Built for cheap local testing (uses `gpt-3.5-turbo` + `text-embedding-3-small`) and easy swap-outs.

## ✨ Features
- **Chroma** vector store (persistent `./.chroma`)
- **Guardrail Agent** filters irrelevant chunks
- **Generator Agent** answers *only* from context
- **Evaluator Agent** returns JSON `{ score, explanation }`
- **Ingestion** for `.txt`/`.md` with token-aware chunking
- **Pretty logs** via `rich`
- Simple CLI: `python main.py "your question"`

## 🧱 Structure
