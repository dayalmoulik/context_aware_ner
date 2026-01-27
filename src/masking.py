from typing import List, Tuple


def mask_entities(
    token_label_pairs: List[Tuple[str, str]],
    mask_token: str = "[MASK]",
) -> str:
    """
    Masks tokens with BIO labels and reconstructs text.
    """

    masked_tokens = []
    skip = False

    for token, label in token_label_pairs:

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if label.startswith("B-"):
            masked_tokens.append(mask_token)
            skip = True
            continue

        if label.startswith("I-") and skip:
            continue

        skip = False

        if token.startswith("##"):
            masked_tokens[-1] += token[2:]
        else:
            masked_tokens.append(token)

    return " ".join(masked_tokens)
