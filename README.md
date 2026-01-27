# Context-Aware Entity Recognition and Sensitivity Masking

## Project Overview

This project implements a **Context-Aware Named Entity Recognition (NER)** system to identify and mask sensitive entities in text using deep learning. Unlike traditional rule-based approaches, the system leverages a fine-tuned **BERT-based transformer model** to detect sensitive information based on linguistic context.

The application supports:
- Token-level BIO tagging
- Context-aware detection of sensitive entities (e.g., names, cities)
- Automatic masking of detected entities
- Real-time and batch processing via a web interface

---

## Folder Structure

```
Context-Aware-NER/
│
├── app/                # Streamlit web application
├── data/
│   └── raw/            # Dataset.csv
├── docs/               # Design & research document, screenshots
├── models/             # Saved fine-tuned model (generated after training)
├── notebooks/          # Optional exploratory notebooks
├── results/            # Evaluation outputs (metrics, confusion matrix)
├── src/                # Core source code
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Prerequisites
- Anaconda (recommended)
- Python 3.10

### Create and Activate Conda Environment

```powershell
conda create -n ner-env python=3.10 -y
conda activate ner-env
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Dataset

The dataset (`Dataset.csv`) is provided as part of the assignment and contains:
- Filled text templates
- Pre-tokenized WordPiece tokens
- BIO-aligned labels at the subword level

Place the dataset at:
```
data/raw/Dataset.csv
```

---

## Training the Model

To fine-tune the BERT-based NER model, run:

```powershell
python src/train.py
```

This will:
- Load and preprocess the dataset
- Fine-tune the BERT model
- Save the trained model to:

```
models/bert_ner/
```

---

## Evaluating the Model

To evaluate performance and generate metrics:

```powershell
python src/evaluate.py
```

Outputs:
- Precision, Recall, and F1-score (printed and saved)
- Confusion matrix saved in:

```
results/confusion_matrix.png
```

---

## Running Inference and Masking

The inference and masking pipeline is implemented in:
- `src/inference.py`
- `src/masking.py`

These modules are used by the web application for real-time predictions.

---

## Running the Web Application

A user-friendly interface is built using **Streamlit**.

To start the application:

```powershell
streamlit run app/app.py
```

Features:
- Real-time text input and masking
- Batch processing via text file upload
- Context-aware masking of sensitive entities

---

## Example

**Input**
```
Please create a PowerPoint presentation for Casey Dietrich in Stephenville.
```

**Output**
```
Please create a PowerPoint presentation for [MASK] in [MASK].
```

---

## Notes

- Model weights are not committed to the repository.
- Training time may vary depending on hardware.
- CPU-based training is supported.

---

## Author

**Moulik Dayal**

---

## License

This project is developed for academic purposes as part of an NLP application assignment.