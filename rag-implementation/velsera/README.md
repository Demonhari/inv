# Velsera – Research-Paper Analysis Pipeline 🧬

Lightweight, **CPU-friendly** workflow that turns a PubMed abstract into:

* **Cancer ↔︎ Non-Cancer** prediction **+ confidence**
* list of **disease mentions** (SciSpaCy)

raw TXT → preprocess → DistilBERT fine-tune → LoRA refine → evaluate
│
└─ FastAPI /predict

yaml
Copy
Edit

---

## 1 · Folder map

| item / folder          | purpose                                             |
|------------------------|-----------------------------------------------------|
| `data/raw/`            | 1 000 abstracts split in **Cancer / Non-Cancer**    |
| `preprocess.py`        | cleans text → `dataset.parquet`, 80/20 JSONL splits |
| `baseline.py`          | DistilBERT full fine-tune (3 ep)                    |
| `train_lora.py`        | LoRA on top (1.2 % params, 5 ep)                    |
| `evaluate.py`          | Accuracy · macro-F1 · confusion matrix              |
| `extract_diseases.py`  | SciSpaCy `en_ner_bc5cdr_md` NER                     |
| `app.py`               | FastAPI — returns label **and soft-max probs**      |
| `tests/`               | 4 tiny pytest checks (pre-process, API, metrics)    |

---

## 2 · Quick-start

```bash
git clone https://github.com/<you>/velsera.git && cd velsera
unzip data/raw/Dataset__1_.zip -d data/raw          # 1 000 mini-papers

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one-off: SciSpaCy disease model (120 MB)
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

python preprocess.py
python baseline.py
python train_lora.py
python evaluate.py                     # writes metrics → output/metrics.json
python extract_diseases.py data/processed/test.jsonl output/diseases.jsonl
uvicorn app:app --reload               # http://127.0.0.1:8000/docs
Example:

bash
Copy
Edit
curl -sX POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"BRCA1 mutation raises breast-cancer risk"}'
json
Copy
Edit
{
  "label": "Cancer",
  "confidence": 0.9999,
  "probabilities": {"Non-Cancer": 0.0001, "Cancer": 0.9999}
}
3 · Training snapshots
script	trainable params	wall-time*	F1 (test)
baseline.py	67.7 M (100 %)	6 min	0.98
train_lora.py	0.81 M (1 %)	7 min	0.99

* 12-core CPU (no GPU). Runs in < 8 GB RAM.

4 · Model choice
backbone	RAM(fp16)	F1	verdict
DistilBERT-base	2.6 GB	0.99	✅ chosen – fast & tiny
microsoft/phi-2 (LoRA-8)	9 GB	0.98	❌ needs big GPU
google/gemma-2b-it	8 GB	0.97	❌ slower, lower F1

DistilBERT satisfies the commodity hardware constraint; swapping models is a one-liner.

5 · Tests & CI
bash
Copy
Edit
pytest -q           # all 4 green
.github/workflows/ci.yml runs tests on every push and builds the Docker image on main.

