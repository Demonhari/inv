import spacy, json, sys, pathlib
from collections import defaultdict

STOP = {
    "cancer", "tumor", "tumors", "tumour", "tumours",
    "infection", "infections", "pain", "death", "toxicity",
    "bleeding", "bleedings", "inflammation"
}

nlp = spacy.load("en_ner_bc5cdr_md")

def extract_clean(txt: str) -> list[str]:
    seen_cui = set()
    nice = []
    for ent in nlp(txt).ents:
        if ent.label_ != "DISEASE":
            continue
        span = ent.text.lower().strip()
        if span in STOP or len(span) < 3 or not any(c.isalpha() for c in span):
            continue

        # UMLS linking – pick the highest-scoring CUI
        if ent._.kb_ents:
            cui, score = ent._.kb_ents[0]
            if score < 0.80:        # you can tighten or loosen this
                continue
            if cui in seen_cui:
                continue
            seen_cui.add(cui)
            nice.append(span)       # or save `cui` if you prefer

        else:                       # fallback, no linker hit
            if span not in nice:
                nice.append(span)
    return nice
