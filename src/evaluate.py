import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForTokenClassification

from src.data_loader import load_dataset, parse_token_columns, train_test_split_data
from src.preprocessing import NERPreprocessor


def strip_bio(label):
    """Remove BIO prefix from NER labels"""
    if label == "O":
        return "O"
    return label.split("-", 1)[-1]


def main():
    # =============================
    # Configuration
    # =============================
    DATA_PATH = "data/raw/Dataset.csv"
    MODEL_DIR = "models/bert_ner"
    MODEL_NAME = "bert-base-cased"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    RESULTS_DIR = "results"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =============================
    # Load dataset
    # =============================
    df = load_dataset(DATA_PATH)
    tokens, labels = parse_token_columns(df)

    _, test_tokens, _, test_labels = train_test_split_data(tokens, labels)

    # =============================
    # Preprocessing
    # =============================
    preprocessor = NERPreprocessor(
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    preprocessor.build_label_mapping(labels)

    test_encodings = preprocessor.encode_batch(test_tokens, test_labels)

    # =============================
    # Dataset wrapper
    # =============================
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            return {key: tensor[idx] for key, tensor in self.encodings.items()}

    test_dataset = NERDataset(test_encodings)

    # =============================
    # Load model
    # =============================
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # =============================
    # Run inference
    # =============================
    all_preds = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # =============================
    # Flatten predictions (ignore padding)
    # =============================
    y_true = []
    y_pred = []

    PAD_ID = -100  # HuggingFace default padding label

    for true_seq, pred_seq in zip(all_labels, all_preds):
        for t, p in zip(true_seq, pred_seq):
            if t != PAD_ID:
                y_true.append(t)
                y_pred.append(p)

    # =============================
    # Convert IDs → Label Names
    # =============================
    id2label = {v: k for k, v in preprocessor.label2id.items()}

    y_true_labels = [id2label[i] for i in y_true]
    y_pred_labels = [id2label[i] for i in y_pred]

    # =============================
    # BIO → Entity-level labels
    # =============================
    y_true_clean = [strip_bio(l) for l in y_true_labels]
    y_pred_clean = [strip_bio(l) for l in y_pred_labels]

    unique_labels = sorted(list(set(y_true_clean)))
    unique_labels = [l for l in unique_labels if l != "O"]  # exclude O
    
    # =============================
    # Classification Report (Entity-level)
    # =============================
    report = classification_report(
        y_true_clean,
        y_pred_clean,
        labels=unique_labels,
        target_names=unique_labels,
        digits=4,
        zero_division=0
    )

    print("\n===== NER Classification Report (Entity-level) =====\n")
    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report_entity_level.txt"), "w") as f:
        f.write(report)

    # =============================
    # Select Top-N Frequent Labels
    # =============================
    label_counts = Counter(y_true_clean)
    top_labels = [label for label, _ in label_counts.most_common(15)]

    mask = np.isin(y_true_clean, top_labels) & np.isin(y_pred_clean, top_labels)
    y_true_top = np.array(y_true_clean)[mask]
    y_pred_top = np.array(y_pred_clean)[mask]

    # =============================
    # Confusion Matrix (Normalized)
    # =============================
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=top_labels,
        yticklabels=top_labels,
    )
    plt.title("Normalized NER Confusion Matrix (Top-15 Entities)", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"))
    plt.show()

    # =============================
    # Error-Only Confusion Matrix
    # =============================
    cm_err = cm.copy()
    np.fill_diagonal(cm_err, 0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_err,
        cmap="Reds",
        xticklabels=top_labels,
        yticklabels=top_labels,
    )
    plt.title("NER Confusion Matrix (Errors Only)", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_errors.png"))
    plt.show()

    # =============================
    # Top Confused Entity Pairs
    # =============================
    print("\n===== Top NER Confusions =====")
    errors = []

    for i, true_label in enumerate(top_labels):
        for j, pred_label in enumerate(top_labels):
            if i != j and cm[i, j] > 0:
                errors.append((true_label, pred_label, cm[i, j]))

    errors = sorted(errors, key=lambda x: -x[2])[:15]

    for t, p, c in errors:
        print(f"{t} → {p} : {c}")

    # =============================
    # Entity-wise F1 Score Plot
    # =============================
    report_dict = classification_report(
        y_true_clean, y_pred_clean, output_dict=True, zero_division=0
    )

    import pandas as pd
    df_report = pd.DataFrame(report_dict).T.drop("accuracy")

    plt.figure(figsize=(12, 6))
    df_report["f1-score"].sort_values(ascending=False).plot(kind="bar")
    plt.title("NER Entity-wise F1 Score")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "entity_f1_scores.png"))
    plt.show()

    print("\n✅ Evaluation complete. Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
