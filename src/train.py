import os
import torch
from transformers import Trainer, TrainingArguments

from src.data_loader import load_dataset, parse_token_columns, train_test_split_data
from src.preprocessing import NERPreprocessor
from src.model import load_ner_model


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    DATA_PATH = "data/raw/Dataset.csv"
    MODEL_NAME = "bert-base-cased"
    OUTPUT_DIR = "models/bert_ner"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = load_dataset(DATA_PATH)
    tokens, labels = parse_token_columns(df)

    train_tokens, test_tokens, train_labels, test_labels = train_test_split_data(
        tokens, labels
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    preprocessor = NERPreprocessor(
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )
    preprocessor.build_label_mapping(train_labels)

    train_encodings = preprocessor.encode_batch(train_tokens, train_labels)
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

    train_dataset = NERDataset(train_encodings)
    test_dataset = NERDataset(test_encodings)

    # -----------------------------
    # Model
    # -----------------------------
    model = load_ner_model(
        model_name=MODEL_NAME,
        label2id=preprocessor.label2id,
        id2label=preprocessor.id2label,
    )

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="results/logs",
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=preprocessor.tokenizer,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # Save model & tokenizer
    # -----------------------------
    trainer.save_model(OUTPUT_DIR)
    preprocessor.tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete. Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
