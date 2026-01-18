from transformers import AutoModelForTokenClassification
from typing import Dict


def load_ner_model(
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
):
    """
    Load a BERT-based model for token classification (NER).
    """

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    return model
