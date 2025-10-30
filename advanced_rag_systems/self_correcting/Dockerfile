FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

# expect key at runtime
# docker run -e OPENAI_API_KEY=sk-... -v $(pwd)/data:/app/data <image> python ingest.py
CMD ["python", "main.py", "what is a self correcting rag pipeline?"]
