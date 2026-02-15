from typing import List, Tuple
import itertools

# Color palette (automatically assigned)
COLOR_PALETTE = [
    "#3B82F6",  # Blue
    "#10B981",  # Green
    "#EF4444",  # Red
    "#F59E0B",  # Amber
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#F97316",  # Orange
]


def reconstruct_tokens(token_label_pairs: List[Tuple[str, str]]):
    """
    Merge WordPiece tokens properly.
    Returns list of (word, label).
    """
    words = []
    current_word = ""
    current_label = None

    for token, label in token_label_pairs:

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append((current_word, current_label))
            current_word = token
            current_label = label

    if current_word:
        words.append((current_word, current_label))

    return words


# -----------------------------------------
# Masking
# -----------------------------------------
def mask_entities(token_label_pairs: List[Tuple[str, str]]):

    words = reconstruct_tokens(token_label_pairs)
    masked_output = []
    skip = False

    for word, label in words:

        if label and label.startswith("B-"):
            masked_output.append("[MASK]")
            skip = True
        elif label and label.startswith("I-") and skip:
            continue
        else:
            skip = False
            masked_output.append(word)

    return " ".join(masked_output)


# -----------------------------------------
# Highlighting (Dynamic for All Entities)
# -----------------------------------------
def reconstruct_with_highlight(token_label_pairs: List[Tuple[str, str]]):

    words = reconstruct_tokens(token_label_pairs)

    # Extract unique entity types dynamically
    entity_types = sorted({
        label.split("-")[1]
        for _, label in words
        if label and label.startswith("B-")
    })

    # Assign colors dynamically
    color_map = dict(zip(entity_types, itertools.cycle(COLOR_PALETTE)))

    html_output = ""
    current_entity = None
    entity_buffer = []

    for word, label in words:

        if label and label.startswith("B-"):
            if entity_buffer:
                color = color_map.get(current_entity, "#6B7280")
                entity_text = " ".join(entity_buffer)
                html_output += (
                    f"<span style='background-color:{color};"
                    f"color:white;padding:4px 6px;border-radius:6px;'>"
                    f"{entity_text}</span> "
                )
                entity_buffer = []

            current_entity = label.split("-")[1]
            entity_buffer.append(word)

        elif label and label.startswith("I-") and current_entity:
            entity_buffer.append(word)

        else:
            if entity_buffer:
                color = color_map.get(current_entity, "#6B7280")
                entity_text = " ".join(entity_buffer)
                html_output += (
                    f"<span style='background-color:{color};"
                    f"color:white;padding:4px 6px;border-radius:6px;'>"
                    f"{entity_text}</span> "
                )
                entity_buffer = []
                current_entity = None

            html_output += word + " "

    if entity_buffer:
        color = color_map.get(current_entity, "#6B7280")
        entity_text = " ".join(entity_buffer)
        html_output += (
            f"<span style='background-color:{color};"
            f"color:white;padding:4px 6px;border-radius:6px;'>"
            f"{entity_text}</span> "
        )

    return html_output.strip(), color_map
