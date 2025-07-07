#!/usr/bin/env python3
import json
import spacy
nlp = spacy.load("en_ner_bc5cdr_md")

def extract_diseases_from_text(text):
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ == "DISEASE"})

if __name__ == "__main__":
    import sys
    abstracts_file = sys.argv[1]
    output_file = sys.argv[2]
    out = []
    with open(abstracts_file) as f:
        for line in f:
            obj = json.loads(line)
            diseases = extract_diseases_from_text(obj["abstract"])
            out.append({"abstract_id": obj["abstract_id"], "extracted_diseases": diseases})
    with open(output_file, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\\n")
    print(f"Disease extraction written to {output_file}")
