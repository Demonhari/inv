# Velsera Research Paper Classification Pipeline

## Setup
\`\`\`bash
git clone <repo-url>
cd velsera
unzip data/raw/Dataset__1_.zip -d data/raw
pip install -r requirements.txt
\`\`\`

## Usage
1. **Preprocess**  
   \`\`\`bash
   python preprocess.py
   \`\`\`
2. **Baseline Training**  
   \`\`\`bash
   python baseline.py
   \`\`\`
3. **LoRA Fine-Tuning**  
   \`\`\`bash
   python train_lora.py
   \`\`\`
4. **Extract Diseases**  
   \`\`\`bash
   python extract_diseases.py data/processed/test.jsonl output/diseases.jsonl
   \`\`\`
5. **Evaluate**  
   \`\`\`bash
   python evaluate.py
   \`\`\`
6. **Serve API**  
   \`\`\`bash
   uvicorn app:app --reload
   \`\`\`
