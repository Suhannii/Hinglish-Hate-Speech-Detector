"""
model.py
--------
Defines the HuggingFace transformer model for binary hate speech classification.
We use AutoModelForSequenceClassification which adds a classification head
on top of any pretrained transformer (BERT, mBERT, IndicBERT, etc.).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class HateSpeechClassifier(nn.Module):
    """
    Thin wrapper around HuggingFace AutoModelForSequenceClassification.
    num_labels=2 → binary classification (0=non-hate, 1=hate)
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased", num_labels: int = 2):
        super().__init__()
        print(f"[model] Loading pretrained model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model_name = model_name

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass.
        If labels are provided, the HuggingFace model automatically computes
        cross-entropy loss and returns it alongside logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return outputs  # outputs.loss, outputs.logits

    def save(self, path: str):
        """Save model weights and config."""
        self.model.save_pretrained(path)
        print(f"[model] Model saved to '{path}'")

    @classmethod
    def load(cls, path: str, num_labels: int = 2):
        """Load a previously saved model."""
        instance = cls.__new__(cls)
        super(HateSpeechClassifier, instance).__init__()
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels
        )
        instance.model_name = path
        print(f"[model] Model loaded from '{path}'")
        return instance


def get_device() -> torch.device:
    """Returns GPU if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] Using device: {device}")
    return device


if __name__ == "__main__":
    # Quick sanity check — loads model and prints parameter count
    clf = HateSpeechClassifier()
    total_params = sum(p.numel() for p in clf.model.parameters())
    print(f"Total parameters: {total_params:,}")
