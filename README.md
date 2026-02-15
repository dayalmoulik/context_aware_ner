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
│   .gitignore
│   LICENSE
│   nlpa_env.yml
│   README.md
│   requirements.txt
│   run.ps1
│   __init__.py
│
├───app
│   │   app.py
│   │   utils.py
│   │
│   └───__pycache__
│           app.cpython-310.pyc
│
├───data
│   ├───processed
│   └───raw
│           Dataset.csv
│
├───docs
│   └───Screenshots
│           01_dataset_preview.png
│
├───models
│   ├───bert_ner
│   │   │   config.json
│   │   │   model.safetensors
│   │   │   special_tokens_map.json
│   │   │   tokenizer.json
│   │   │   tokenizer_config.json
│   │   │   training_args.bin
│   │   │   vocab.txt
│   │   │
│   │   └───checkpoint-5928
│   │           config.json
│   │           model.safetensors
│   │           optimizer.pt
│   │           rng_state.pth
│   │           scheduler.pt
│   │           special_tokens_map.json
│   │           tokenizer.json
│   │           tokenizer_config.json
│   │           trainer_state.json
│   │           training_args.bin
│   │           vocab.txt
│   │
│   └───distilbert_ner
│       │   config.json
│       │   model.safetensors
│       │   special_tokens_map.json
│       │   tokenizer.json
│       │   tokenizer_config.json
│       │   training_args.bin
│       │   vocab.txt
│       │
│       └───checkpoint-5928
│               config.json
│               model.safetensors
│               optimizer.pt
│               rng_state.pth
│               scheduler.pt
│               special_tokens_map.json
│               tokenizer.json
│               tokenizer_config.json
│               trainer_state.json
│               training_args.bin
│               vocab.txt
│
├───notebooks
├───results
│       classification_report_entity_level.txt
│       confusion_matrix_errors.png
│       confusion_matrix_normalized.png
│       entity_f1_scores.png
│
└───src
    │   config.py
    │   data_loader.py
    │   evaluate.py
    │   inference.py
    │   masking.py
    │   model.py
    │   preprocessing.py
    │   train.py
    │   __init__.py
    │
    └───__pycache__
            data_loader.cpython-310.pyc
            evaluate.cpython-310.pyc
            inference.cpython-310.pyc
            masking.cpython-310.pyc
            model.cpython-310.pyc
            preprocessing.cpython-310.pyc
            train.cpython-310.pyc
            train.cpython-313.pyc
            __init__.cpython-310.pyc
            __init__.cpython-313.pyc
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
or run as module:

```powershell
python -m src.train
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

or run as module:

```powershell
python -m src.evaluate
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