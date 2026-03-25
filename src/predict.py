"""
predict.py
----------
Inference module.
Loads a saved model and tokenizer, then exposes a predict(text) function
that returns the label and confidence score for any Hinglish sentence.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import clean_text
from src.model import get_device

# Label mapping
LABEL_MAP = {0: "Non-Hate", 1: "Hate"}


class HateSpeechPredictor:
    """
    Loads a fine-tuned model from disk and provides a predict() method.
    """

    def __init__(self, model_dir: str = "models/saved_model",
                 base_model: str = "bert-base-multilingual-cased",
                 max_length: int = 128):
        self.device = get_device()
        self.max_length = max_length

        print(f"[predict] Loading tokenizer from base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        print(f"[predict] Loading fine-tuned model from: {model_dir}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("[predict] Model ready.")

    def predict(self, text: str) -> dict:
        """
        Predicts hate speech for a single Hinglish sentence.

        Args:
            text: Raw Hinglish string

        Returns:
            dict with keys:
              - original_text  : the input as-is
              - cleaned_text   : after preprocessing
              - label          : "Hate" or "Non-Hate"
              - label_id       : 1 or 0
              - confidence     : float 0–1 (probability of predicted class)
              - probabilities  : {"Non-Hate": float, "Hate": float}
        """
        cleaned = clean_text(text)

        encoding = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        probs = F.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        label_id = int(probs.argmax())
        confidence = float(probs[label_id])

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "label": LABEL_MAP[label_id],
            "label_id": label_id,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Non-Hate": round(float(probs[0]) * 100, 2),
                "Hate": round(float(probs[1]) * 100, 2),
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Runs predict() on a list of texts and returns a list of result dicts."""
        return [self.predict(t) for t in texts]


def demo_predictions(predictor: HateSpeechPredictor):
    """Runs a few demo predictions and prints them nicely."""
    demo_texts = [
        "Yeh log bahut gande hain, inhe yahan se nikalo",
        "Aaj mausam bahut achha hai, bahar jaana chahiye",
        "Tum log kisi kaam ke nahi ho, chale jao",
        "Mujhe chai bahut pasand hai, especially masala chai",
        "Is community ke log sab chor hote hain",
        "Yeh movie bahut achi thi, tumhe dekhni chahiye",
    ]

    print("\n" + "=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)
    for text in demo_texts:
        result = predictor.predict(text)
        print(f"\nText      : {result['original_text']}")
        print(f"Prediction: {result['label']}  (confidence: {result['confidence']}%)")
        print(f"Probs     : Non-Hate={result['probabilities']['Non-Hate']}%  "
              f"Hate={result['probabilities']['Hate']}%")
    print("=" * 60)


if __name__ == "__main__":
    predictor = HateSpeechPredictor()
    demo_predictions(predictor)
