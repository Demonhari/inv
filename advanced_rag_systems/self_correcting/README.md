# Self-Correcting RAG (OpenAI + Chroma)

A modular RAG pipeline with **Retriever → Guardrail → Generator → Evaluator → Feedback** agents.  
Built for cheap local testing (uses `gpt-3.5-turbo` + `text-embedding-3-small`) and designed to scale easily.

---

## ✨ Features
- **Chroma** vector store (persistent `./.chroma`)
- **Guardrail Agent** filters irrelevant chunks
- **Generator Agent** answers *only* from context
- **Evaluator Agent** returns JSON `{ "score": <int>, "explanation": "<text>" }`
- **Feedback Agent** improves future answers automatically
- **Ingestion** for `.txt`/`.md` with token-aware chunking
- **Pretty logs** via `rich`
- **FastAPI Server** with `/ask`, `/metrics`, and `/metrics/prom`
- **LangSmith tracing** + **Prometheus metrics**
- **Dockerfile**, **Makefile**, and **CI workflow** for production deployment

---

## 🧱 Structure

## 📊 Observability & Metrics

- **LangSmith Tracing**: Set these env vars to enable:
  ```bash
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY=lsv2_***
  export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
  export LANGCHAIN_PROJECT=self-correcting-rag
