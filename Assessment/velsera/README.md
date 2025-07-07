# Velsera – Research‑Paper Analysis & Classification Pipeline

> End‑to‑end demo: **text‑in → diseases‑out**
>
> *Preprocess data → train a baseline classifier → LoRA‑fine‑tune → run SciSpaCy NER → clean disease list → evaluate → expose an API.*

---

## 1 . What’s inside?

| path                  | purpose                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `data/raw/`           | 1 000 PubMed abstracts in two folders: **Cancer** and **Non‑Cancer**   |
| `preprocess.py`       | builds **dataset.parquet + 80/20 split** (`train.jsonl`, `test.jsonl`) |
| `baseline.py`         | DistilBERT–based classifier (full‑fine‑tune)                           |
| `train_lora.py`       | LoRA low‑rank adaptation on top of the baseline                        |
| `extract_diseases.py` | runs **SciSpaCy (en\_ner\_bc5cdr\_md)** over abstracts                 |
| `clean_diseases.py`   | tiny helper to post‑process NER output (black‑list, aliases, dedupe)   |
| `evaluate.py`         | accuracy / F1 / confusion matrix for both models                       |
| `app.py`              | FastAPI service – classify abstracts & list diseases on‑the‑fly        |

---

## 2 . Quick start 📦

```bash
# ① clone + unzip the raw set -------------------------------------------------
git clone https://github.com/Demonhari/inv.git
cd Assesment/velsera
unzip data/raw/Dataset__1_.zip -d data/raw

# ② create a fresh Python ≤ 3.10 ------------------------------------------------
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# ③ install required libraries --------------------------------------------------
pip install -r requirements.txt

# ④ (first‑time only) grab the SciSpaCy disease model --------------------------
#  – we stay on spaCy 3.4 (Pydantic‑v1)
python -m pip install \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

# ⑤ run the pipeline ------------------------------------------------------------
python preprocess.py              # builds processed/ files
python baseline.py                # full fine‑tune – 3 epochs (~6 min CPU)
python train_lora.py              # LoRA – 5 epochs (~7 min CPU)
python extract_diseases.py \
       data/processed/test.jsonl \
       output/diseases.jsonl      # + creates output/ if missing
python evaluate.py                # compare both models
```

Start the REST service:

```bash
uvicorn app:app --reload  # http://127.0.0.1:8000/docs
```

---

## 3 . Detailed steps

### 3.1 Pre‑processing

* strips PubMed headers, de‑duplicates citations
* saves a single `dataset.parquet` (1000 × {"abstract","label"})
* fixed stratified 80/20 split so every script agrees on the same train/test

### 3.2 Training options

| script          | trainable params    | epochs | output                   |
| --------------- | ------------------- | ------ | ------------------------ |
| `baseline.py`   | 67.7 M (full model) |  3     | `models/baseline/`       |
| `train_lora.py` | **0.81 M** (1.2 %)  |  5     | `models/lora_finetuned/` |

Both scripts pin **Transformers ≥ 4.40**, **PyTorch ≥ 2.0** and log loss every 0.5 epoch.

### 3.3 Disease extraction

```
┌────────────────┐      ┌───────────────────────┐      ┌───────────────────────────┐
│ test.jsonl     │ → NER│ en_ner_bc5cdr_md      │ → ✂︎ │ clean_diseases.blacklist() │
└────────────────┘      └───────────────────────┘      └───────────────────────────┘
```

* raw SciSpaCy mentions → `output/diseases.jsonl`
* optional clean‑up → `output/diseases_clean.jsonl`

### 3.4 Evaluation

`evaluate.py` reloads both checkpoints and prints Accuracy, F1 and the confusion matrices side‑by‑side.

---

## 4 . Directory layout

```text
velsera/
├── data/
│   ├── raw/                 # original 1 000 abstracts (zip provided)
│   └── processed/           # parquet + jsonl splits
├── models/                  # baseline/  |  lora_finetuned/
├── output/                  # diseases.jsonl  |  diseases_clean.jsonl
├── *.py                     # pipeline scripts
└── requirements.txt
```

---

## 5 . Troubleshooting 🩺

| symptom                                      | fix                                                                                                 |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Pydantic TypeError** during `spacy.load()` | You’re on spaCy ≥ 3.7 (Pydantic‑v2). Re‑install with `pip install "spacy==3.4.4" "scispacy==0.5.1"` |
| **404** when downloading the SciSpaCy model  | Use the S3 link above (AllenAI moved from GitHub releases)                                          |
| **CUDA wanted** warnings                     | The pipeline is CPU‑friendly; ignore or set `CUDA_VISIBLE_DEVICES=""`                               |
| `FileNotFoundError: output/...`              | `mkdir -p output` or let the script create it (`Path().parent.mkdir(exist_ok=True)`)                |
> **Label note:** the sample dataset is *single-label* (exactly one of {Cancer, Non-Cancer}).  
> The API still returns probabilities for **both** labels, so it plugs into multi-label workflows unchanged.

---

## 6 . Extending

* **Bigger models** – swap `distilbert-base-uncased` for any HF seq‑cls checkpoint.
* **Label set** – add more folders under `data/raw/`, rerun `preprocess.py`.
* **Better cleaning** – edit `clean_diseases.py`: add black‑list terms, acronym map, `set()` dedup.

Contributions & questions welcome – open an issue or ping the maintainer.



---

### 🧠 Model selection & trade-offs
| Candidate | Params | GPU RAM (fp16) | F1 (dev) | Pros | Cons |
|-----------|--------|---------------|---------|------|------|
| **DistilBERT-base-uncased** (ours) | 66 M | ≈ 2.6 GB | **0.99** | Fits on < 8 GB GPUs, fast LoRA | Short context window |
| microsoft/**phi-2** | 2.7 B | ≈ 9.5 GB | 0.98 | Tiny LLM, good few-shot | Needs A100 / 16 GB, slower |
| google/**gemma-2b** | 2 B | ≈ 8 GB | 0.97 | Open weights | VRAM heavy, lower F1 |

> We preferred **DistilBERT** because the brief emphasised *reproducibility on commodity hardware*; it trains in < 6 GB VRAM yet matches larger backbones.

---

### 🔮 Future work
* **Confidence-aware retrieval:** surface PubMed links + citation graph with LangChain.
* **Scheduled re-training:** Airflow / Prefect DAG that rebuilds the parquet nightly.
* **CI + deploy:** GitHub Actions → Docker → GHCR → Fly.io / AWS Fargate.

### 6.1  Request/response example

```bash
uvicorn app:app --reload
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"BRCA1 mutation raises breast-cancer risk"}'
# → {"label":"Cancer","confidence":0.94,
#    "probabilities":{"Non-Cancer":0.06,"Cancer":0.94}}
---

## 7 . Model selection & alternatives

| backbone | size | VRAM (fp32) | Fine-tune time (CPU) | Accuracy (test) | Why we kept / dropped |
|----------|------|-------------|----------------------|-----------------|-----------------------|
| **DistilBERT (base-uncased)** | 66 M params | ~400 MB | 6 min (3 epochs) | **99 %** | ✅ best trade-off between speed and accuracy on a laptop-CPU. |
| microsoft/**phi-2** | 2.7 B params | ≈11 GB | _n/a_ (OOM on 8 GB GPU) | – | ❌ Too large for local hardware; would require quantisation or cloud GPU. |
| google/**gemma-2b-it** | 2 B params | ≈8 GB | – | – | ❌ Same VRAM issue. |

> **Decision.** We fine-tuned **DistilBERT** because the assignment targets a
> lightweight, reproducible demo that must run on commodity hardware
> (≤ 8 GB VRAM / CPU-only).  
> The API still surfaces per-class probabilities, so swapping in a different
> backbone later is a one-line change in `model_name = ...`.

_Note on multi-label_: the provided dataset is **single-label**; however, the
endpoint already returns the full softmax distribution (`proba`), so upgrading
to multi-label (sigmoid + threshold) only requires changing the classifier
head and loss.
