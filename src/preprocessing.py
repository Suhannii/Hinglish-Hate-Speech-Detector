"""
preprocessing.py
----------------
Handles all text cleaning and tokenization for Hinglish text.
Steps:
  1. Lowercase
  2. Remove URLs
  3. Remove emojis and special characters
  4. Basic Hinglish normalization (common abbreviations)
  5. Tokenization via HuggingFace tokenizer
"""

import re
import pandas as pd
from transformers import AutoTokenizer

# ── Common Hinglish abbreviations / normalizations ────────────────────────────
HINGLISH_NORM = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "m": "main",
    "h": "hai",
    "k": "ka",
    "n": "nahi",
    "b": "bhi",
    "d": "de",
    "2": "to",
    "4": "for",
    "gr8": "great",
    "lol": "haha",
    "btw": "by the way",
    "idk": "i don't know",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "omg": "oh my god",
    "wtf": "what the",
    "bc": "because",
    "coz": "because",
    "kya": "kya",
    "nahi": "nahi",
    "hai": "hai",
    "hain": "hain",
}


def clean_text(text: str) -> str:
    """
    Cleans a single Hinglish text string.
    Returns the cleaned string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http/https/www links)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove emojis (unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # 4. Remove special characters but keep Devanagari script and basic punctuation
    #    Keep: a-z, 0-9, spaces, and Devanagari Unicode block (U+0900–U+097F)
    #    Also keep apostrophes so words like "don't" stay intact
    text = re.sub(r"[^a-z0-9\s'\u0900-\u097F]", " ", text)

    # 5. Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Basic Hinglish word normalization
    words = text.split()
    words = [HINGLISH_NORM.get(w, w) for w in words]
    text = " ".join(words)

    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Applies clean_text to every row in the dataframe.
    Adds a 'clean_text' column.
    """
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    # Drop rows where cleaning resulted in empty string
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    print(f"[preprocessing] Cleaned {len(df)} samples.")
    return df


class HinglishTokenizer:
    """
    Wraps a HuggingFace tokenizer for easy batch tokenization.
    Default model: bert-base-multilingual-cased (supports Hindi + English)
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased", max_length: int = 128):
        print(f"[preprocessing] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize(self, texts: list) -> dict:
        """
        Tokenizes a list of strings.
        Returns a dict with input_ids, attention_mask, token_type_ids.
        """
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def tokenize_single(self, text: str) -> dict:
        """Tokenizes a single string."""
        return self.tokenize([text])


if __name__ == "__main__":
    # Quick test
    sample = "Yeh log bahut gande hain!! Check http://example.com 😡"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
