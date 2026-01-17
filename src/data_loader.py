import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load dataset CSV containing tokenized text and BIO labels.
    """
    df = pd.read_csv(csv_path)
    return df


def parse_token_columns(df: pd.DataFrame) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Parses 'Tokenised Filled Template' and 'Tokens' columns.
    These are stored as string representations of Python lists.
    """

    if not {"Tokenised Filled Template", "Tokens"}.issubset(df.columns):
        raise ValueError(
            "Expected columns: 'Tokenised Filled Template' and 'Tokens'"
        )

    tokens = df["Tokenised Filled Template"].apply(eval).tolist()
    labels = df["Tokens"].apply(eval).tolist()

    return tokens, labels


def train_test_split_data(
    tokens: List[List[str]],
    labels: List[List[str]],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split token and label sequences into train and test sets.
    """
    return train_test_split(
        tokens,
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


if __name__ == "__main__":
    df = load_dataset("data/raw/Dataset.csv")
    tokens, labels = parse_token_columns(df)

    print("Number of samples:", len(tokens))
    print("Sample tokens:", tokens[0])
    print("Sample labels:", labels[0])
