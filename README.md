"# spycy_ner_custom_model" 
# 🧠 CV Skill Extraction using spaCy NER

This project uses spaCy to train a custom Named Entity Recognition (NER) model for extracting **skills** from resumes or CV text.

---

## 📁 Project Structure

```
.
├── config.cfg                  # spaCy training config
├── cv_skills_dataset.spacy    # Converted training data (DocBin format)
├── convert.py                 # Script to convert JSON to spaCy format
├── output/                    # Folder where trained model will be saved
└── README.md
```

---

## ⚙️ Setup Instructions

### ✅ 1. Install Dependencies

```bash
pip install -U spacy tqdm
python -m spacy download en_core_web_lg
```

---

### ✅ 2. Prepare Your Data

Example format (`cv_skills.json`):

```json
[
  {
    "text": "Skilled in Python, Docker and Kubernetes.",
    "entities": [[11, 17, "SKILL"], [19, 25, "SKILL"], [30, 40, "SKILL"]]
  }
]
```

Convert it using `convert.py`:

```python
# convert.py
import spacy
from spacy.tokens import DocBin
import json

nlp = spacy.blank("en")
doc_bin = DocBin()

with open("cv_skills.json", "r", encoding="utf8") as f:
    data = json.load(f)

for record in data:
    text = record["text"]
    ents = record["entities"]
    doc = nlp.make_doc(text)
    spans = []
    for start, end, label in ents:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            spans.append(span)
    doc.ents = spans
    doc_bin.add(doc)

doc_bin.to_disk("cv_skills_dataset.spacy")
```

Run the converter:

```bash
python convert.py
```

---

### ✅ 3. Generate spaCy Config

```bash
python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency
```

Open `config.cfg` and set:

```ini
[paths]
train = "cv_skills_dataset.spacy"
dev = "cv_skills_dataset.spacy"
vectors = "en_core_web_lg"
```

---

### ✅ 4. Train the Model (CPU)

```bash
python -m spacy train config.cfg --output output --gpu-id -1
```

(Optional GPU: `--gpu-id 0` if CuPy is installed)

---

## 🧪 Inference Example

```python
import spacy

nlp = spacy.load("output/model-best")
doc = nlp("Proficient in JavaScript, React, and Docker.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

## ✅ Notes & Troubleshooting

- Error `[E913] Corpus path can't be None` → make sure `paths.train` and `paths.dev` are set in `config.cfg`.
- Error `[E884] Vectors could not be found` → run: `python -m spacy download en_core_web_lg`.
- JSON must be a list of dictionaries. Invalid: `{}`. Valid: `[{}, {}, ...]`.

---

## 📄 License

MIT License
