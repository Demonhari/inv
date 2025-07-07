#!/usr/bin/env python3
import json, sys, spacy

nlp = spacy.load("en_ner_bc5cdr_md")

def extract_diseases(text: str):
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ == "DISEASE"})

if __name__ == "__main__":
    abstracts_file = sys.argv[1]
    output_file    = sys.argv[2]

    out = []
    with open(abstracts_file) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            diseases = extract_diseases(obj["abstract"])
            rec_id   = obj.get("abstract_id", idx)   # fallback to line index
            out.append({"abstract_id": rec_id, "extracted_diseases": diseases})

    with open(output_file, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")

    print(f"Disease extraction written to {output_file}")
