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

        input_ids = []
        attention_masks = []
        label_ids = []

        for token_seq, label_seq in zip(tokens, labels):

            # Convert tokens to IDs using tokenizer vocab
            ids = self.tokenizer.convert_tokens_to_ids(token_seq)

            # Truncate
            ids = ids[: self.max_length]
            label_seq = label_seq[: self.max_length]

            # Attention mask
            mask = [1] * len(ids)

            # Padding
            padding_len = self.max_length - len(ids)
            ids += [self.tokenizer.pad_token_id] * padding_len
            mask += [0] * padding_len
            label_seq += ["O"] * padding_len

            # Map labels to IDs
            label_ids_seq = [self.label2id[label] for label in label_seq]

            input_ids.append(ids)
            attention_masks.append(mask)
            label_ids.append(label_ids_seq)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(label_ids),
        }
