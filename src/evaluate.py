import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForTokenClassification

from src.data_loader import load_dataset, parse_token_columns, train_test_split_data
from src.preprocessing import NERPreprocessor


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    DATA_PATH = "data/raw/Dataset.csv"
    MODEL_DIR = "models/bert_ner"
    MODEL_NAME = "bert-base-cased"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    RESULTS_DIR = "results"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = load_dataset(DATA_PATH)
    tokens, labels = parse_token_columns(df)

    _, test_tokens, _, test_labels = train_test_split_data(
        tokens, labels
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    preprocessor = NERPreprocessor(
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    preprocessor.build_label_mapping(labels)

    test_encodings = preprocessor.encode_batch(test_tokens, test_labels)

    # -----------------------------
    # Dataset wrapper
    # -----------------------------
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            return {
                key: tensor[idx]
                for key, tensor in self.encodings.items()
            }

    test_dataset = NERDataset(test_encodings)

    # -----------------------------
    # Load model
    # -----------------------------
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # -----------------------------
    # Run inference
    # -----------------------------
    all_preds = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE
    )

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

    # -----------------------------
    # Flatten predictions (ignore padding)
    # -----------------------------
    y_true = []
    y_pred = []

    for true_seq, pred_seq in zip(all_labels, all_preds):
        for t, p in zip(true_seq, pred_seq):
            if t != preprocessor.label2id["O"] or p != preprocessor.label2id["O"]:
                y_true.append(t)
                y_pred.append(p)

    label_names = list(preprocessor.label2id.keys())

    # -----------------------------
    # Classification report
    # -----------------------------
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4,
    )

    print(report)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        xticklabels=label_names,
        yticklabels=label_names,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("NER Confusion Matrix")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.show()

    print("Evaluation complete. Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
