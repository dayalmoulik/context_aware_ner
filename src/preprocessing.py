import encodings
from typing import List, Dict
import torch
from transformers import AutoTokenizer


class NERPreprocessor:
    def __init__(
        self,
        model_name: str = "bert-base-cased",
        max_length: int = 128,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

    def build_label_mapping(self, labels: List[List[str]]):
        """
        Build label2id and id2label mappings from dataset labels.
        """
        unique_labels = sorted({label for seq in labels for label in seq})
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def encode_batch(
        self,
        tokens: List[List[str]],
        labels: List[List[str]],
    ):
        """
        Convert tokens and labels into BERT-compatible tensors.
        Assumes tokens are already WordPiece-tokenized.
        """

        encodings = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_offsets_mapping=False)

        encoded_labels = []

        for i, label_seq in enumerate(labels):
            word_ids = encodings.word_ids(batch_index=i)
            label_ids = []

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)        # special tokens
                elif word_id >= len(label_seq):
                # 🔑 CRITICAL FIX: truncated words
                    label_ids.append(-100)
                else:
                    label_ids.append(self.label2id[label_seq[word_id]])
            
            encoded_labels.append(label_ids)

        return {
        "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encodings["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(encoded_labels, dtype=torch.long),
        }