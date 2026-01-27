import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple


class NERInference:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform NER inference on raw text.
        Returns list of (token, label).
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        labels = [
            self.model.config.id2label[pred] for pred in predictions
        ]

        return list(zip(tokens, labels))
