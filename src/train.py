"""
train.py
--------
Handles dataset splitting, PyTorch Dataset/DataLoader creation,
and the full training loop with loss + accuracy tracking.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocessing import preprocess_dataframe, HinglishTokenizer
from src.model import HateSpeechClassifier, get_device


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class HinglishDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Stores tokenized inputs and labels for the DataLoader.
    """

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings  # dict of tensors from tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a single sample as a dict of tensors
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_data(
    csv_path: str,
    model_name: str = "bert-base-multilingual-cased",
    max_length: int = 128,
    test_size: float = 0.2,
    batch_size: int = 16,
):
    """
    Loads CSV → cleans text → tokenizes → splits → returns DataLoaders.
    """
    # Load
    df = pd.read_csv(csv_path)
    print(f"[train] Loaded {len(df)} samples from '{csv_path}'")

    # Clean
    df = preprocess_dataframe(df, text_col="text")

    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    print(f"[train] Train: {len(train_df)} | Test: {len(test_df)}")

    # Tokenize
    tokenizer_wrapper = HinglishTokenizer(model_name=model_name, max_length=max_length)

    train_enc = tokenizer_wrapper.tokenize(train_df["clean_text"].tolist())
    test_enc = tokenizer_wrapper.tokenize(test_df["clean_text"].tolist())

    # Datasets
    train_dataset = HinglishDataset(train_enc, train_df["label"].tolist())
    test_dataset = HinglishDataset(test_enc, test_df["label"].tolist())

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer_wrapper.tokenizer


# ── Training loop ──────────────────────────────────────────────────────────────

def train_model(
    csv_path: str,
    model_name: str = "bert-base-multilingual-cased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    save_dir: str = "models/saved_model",
):
    """
    Full training loop.
    Returns the trained model and history dict (loss/accuracy per epoch).
    """
    device = get_device()

    # Prepare data
    train_loader, test_loader, _ = prepare_data(
        csv_path, model_name=model_name, max_length=max_length, batch_size=batch_size
    )

    # Model
    classifier = HateSpeechClassifier(model_name=model_name)
    classifier.model.to(device)

    # Optimizer
    optimizer = AdamW(classifier.model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler (linear warmup then decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # History for plotting
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"\n[train] Starting training for {epochs} epoch(s)...\n")

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        classifier.model.train()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # ── Validation phase ──
        val_loss, val_acc = evaluate_epoch(classifier, test_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch} → "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    classifier.save(save_dir)

    return classifier, history, test_loader


def evaluate_epoch(classifier, loader, device):
    """Runs one pass over loader and returns avg loss and accuracy."""
    classifier.model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


if __name__ == "__main__":
    train_model(csv_path="data/hinglish_hate_speech.csv")
