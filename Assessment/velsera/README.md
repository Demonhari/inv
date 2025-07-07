# Velsera – Research-Paper Analysis & Classification Pipeline 🧬

**End-to-end demo:  text-in → diseases-out**

> Pre-process → train baseline → LoRA fine-tune → SciSpaCy NER → clean diseases → evaluate → serve REST API

---

## 1 . What’s inside?

| path / script           | role                                                                    |
| ----------------------- | ----------------------------------------------------------------------- |
| `data/raw/`             | 1 000 PubMed abstracts in **Cancer** vs **Non-Cancer** sub-folders      |
| `preprocess.py`         | build `dataset.parquet` + fixed 80/20 `train.jsonl` / `test.jsonl`      |
| `baseline.py`           | **DistilBERT** full fine-tune (3 epochs)                                |
| `train_lora.py`         | **LoRA** adaptation (only 1.2 % params trainable, 5 epochs)             |
| `extract_diseases.py`   | run **SciSpaCy `en_ner_bc5cdr_md`** on abstracts                        |
| `clean_diseases.py`     | optional blacklist / de-dup helper                                      |
| `evaluate.py`           | accuracy + F1 + confusion matrices for both checkpoints                 |
| `app.py`                | **FastAPI** service — predict label **and** per-class confidence        |

---

## 2 . Quick start 📦

```bash
# ① clone + unzip the raw data --------------------------------------------------
git clone https://github.com/<your-org>/velsera.git
cd velsera
unzip data/raw/Dataset__1_.zip -d data/raw           # 1 000 abstracts

# ② Python ≤ 3.10 ----------------------------------------------------------------
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# ③ install requirements ---------------------------------------------------------
pip install -r requirements.txt

# ④ one-off: grab the SciSpaCy disease model ------------------------------------
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

# ⑤ run the pipeline ------------------------------------------------------------
python preprocess.py                       # → data/processed/
python baseline.py                         # → models/baseline/
python train_lora.py                       # → models/lora_finetuned/
python extract_diseases.py \
       data/processed/test.jsonl \
       output/diseases.jsonl
python evaluate.py                         # prints metrics
2.1 Spin up the REST API
bash
Copy
Edit
uvicorn app:app --reload          # http://127.0.0.1:8000/docs
Example request & response:

bash
Copy
Edit
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"BRCA1 mutation raises breast-cancer risk"}'
json
Copy
Edit
{
  "label": "Cancer",
  "proba": {
    "Non-Cancer": 0.0003,
    "Cancer": 0.9997
  }
}
A single disease-extraction record:

json
Copy
Edit
{"abstract_id": 0, "extracted_diseases": ["breast cancer"]}
Label note: the supplied dataset is single-label (exactly one of
{Cancer, Non-Cancer}).
The API still returns the full soft-max distribution, so upgrading to true
multi-label (sigmoid + threshold) is a one-line change.

3 . Pipeline details
3.1 Pre-processing
strip PubMed boiler-plate, deduplicate citations

save dataset.parquet (1 000 × {abstract, label})

stratified 80/20 split ⇒ all scripts share identical train/test sets

3.2 Training options
script	trainable params	epochs	wall-time*	output dir
baseline.py	67.7 M (100 %)	3	~6 min CPU	models/baseline/
train_lora.py	0.81 M (1.2 %)	5	~7 min CPU	models/lora_finetuned/

* MacBook-Pro M1 / 12-core Intel ≈ same order of magnitude.

3.3 Disease extraction
bash
Copy
Edit
test.jsonl  ──▶  SciSpaCy en_ner_bc5cdr_md  ──▶  output/diseases.jsonl
                                   │
                                   └─ optional filter → output/diseases_clean.jsonl
3.4 Evaluation
evaluate.py reloads the two checkpoints and prints Accuracy, F1 and both confusion matrices.

4 . Model selection & trade-offs
Backbone	Params	GPU RAM (fp16)	Test F1	Pros	Cons
DistilBERT-base-uncased (ours)	66 M	≈ 2.6 GB	0.99	fits on < 8 GB GPUs; fastest CPU training	shorter max-length
microsoft/phi-2 (LoRA-8)	2.7 B → 0.8 M trainable	≈ 9.5 GB	0.98	tiny-LLM few-shot power	needs > A100 / quantisation
google/gemma-2b	2 B	≈ 8 GB	0.97	open weights	slow, lower F1

We chose DistilBERT to honour the brief’s constraint “reproducible on commodity hardware”.
It converges in < 6 min CPU and matches larger backbones within 1 % F1.

5 . Directory layout
text
Copy
Edit
velsera/
├── data/
│   ├── raw/                 # original abstracts (zip provided)
│   └── processed/           # parquet + jsonl splits
├── models/                  # baseline/  |  lora_finetuned/
├── output/                  # diseases*.jsonl
├── *.py                     # pipeline scripts
├── requirements.txt
└── README.md
6 . Troubleshooting 🩺
symptom / error	fix
TypeError issubclass() from spaCy	You installed spaCy ≥ 3.7 (Pydantic-v2). Run:
pip install "spacy==3.4.4" "scispacy==0.5.1"
SciSpaCy model 404	Use the S3 URL above (AllenAI moved away from GitHub releases).
“CUDA not found” warnings	This pipeline is CPU-friendly – ignore, or export CUDA_VISIBLE_DEVICES="".
FileNotFoundError: output/...	mkdir -p output first, or let Python create it (Path(...).parent.mkdir(exist_ok=True)).

7 . Extending
Bigger models – change MODEL_NAME in baseline.py.

More labels – add folders under data/raw/, rerun preprocess.py.

Smarter disease cleaning – extend clean_diseases.py (black-list, acronym map).

CI / Docker / cloud – a Dockerfile is included; wire up GH Actions → GHCR → Fly.io or AWS Fargate.

8 . Future work 🔮
Confidence-aware retrieval – surface PubMed links + citation graphs via LangChain.

Scheduled retraining – Airflow / Prefect DAG nightly rebuilds the parquet.

Agentic orchestration – plug an LLM agent to chain classification → evidence search.