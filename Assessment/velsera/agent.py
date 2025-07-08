"""
agent.py
────────────────────────────────────────────────────────────────────────────
User gives a free-text biomedical paragraph.
The agent:
 ① classifies it (internal call)
 ② if Cancer → asks OpenAI to draft a bullet-point summary
    citing the disease list from SciSpaCy.
"""
import os, requests, json
from langchain_openai import ChatOpenAI
from extract_diseases import extract_diseases_spacy  # reuse your helper

PRED_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
llm      = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4)

def classify(text: str):
    r = requests.post(PRED_URL, json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.json()

def agent(text: str):
    cls = classify(text)
    diseases = extract_diseases_spacy(text)

    if cls["label"] == "Non-Cancer":
        return {"classification": cls, "answer": "The abstract is non-cancer."}

    prompt = (
        "You are a biomedical research assistant.\n"
        f"Abstract:\n{text}\n\n"
        f"Predicted label: Cancer (confidence {cls['confidence']:.2%}).\n"
        f"Diseases mentioned: {', '.join(diseases) or 'none'}.\n\n"
        "Write 3 concise bullet points summarising why this is cancer-related. "
        "Cite any disease terms inline."
    )
    answer = llm.invoke(prompt).content
    return {"classification": cls, "answer": answer}

if __name__ == "__main__":
    import sys, pprint
    pprint.pp(agent(sys.argv[1]))
