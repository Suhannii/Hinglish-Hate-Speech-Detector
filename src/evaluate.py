"""
evaluate.py
-----------
Evaluates the trained model on the test set and generates:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix heatmap
  - Loss and accuracy curves
All plots are saved to the 'plots/' directory.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.model import get_device


# ── Metric computation ─────────────────────────────────────────────────────────

def get_predictions(classifier, loader, device):
    """
    Runs inference on the full loader.
    Returns (all_preds, all_labels) as numpy arrays.
    """
    classifier.model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"]

            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def print_metrics(y_true, y_pred):
    """Prints a full classification report."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {acc:.4f}")
    print("\nDetailed Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Non-Hate (0)", "Hate (1)"]
        )
    )
    return acc


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_dir: str = "plots"):
    """Saves a confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Hate", "Hate"],
        yticklabels=["Non-Hate", "Hate"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved to '{path}'")


def plot_training_curves(history: dict, save_dir: str = "plots"):
    """
    Plots loss and accuracy curves from the training history dict.
    history keys: train_loss, val_loss, train_acc, val_acc
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss curve ──
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"[evaluate] Loss curve saved to '{loss_path}'")

    # ── Accuracy curve ──
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"[evaluate] Accuracy curve saved to '{acc_path}'")


def plot_label_distribution(df, save_dir: str = "plots"):
    """Bar chart of class distribution in the dataset."""
    os.makedirs(save_dir, exist_ok=True)
    counts = df["label"].value_counts().sort_index()
    labels = ["Non-Hate (0)", "Hate (1)"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts.values, color=["steelblue", "tomato"], edgecolor="black")
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(count), ha="center", va="bottom", fontweight="bold")
    plt.title("Dataset Label Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(save_dir, "label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Label distribution saved to '{path}'")


def run_evaluation(classifier, test_loader, history: dict, save_dir: str = "plots"):
    """
    Master evaluation function — call this after training.
    Prints metrics and saves all plots.
    """
    device = get_device()
    y_pred, y_true = get_predictions(classifier, test_loader, device)
    acc = print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, save_dir=save_dir)
    plot_training_curves(history, save_dir=save_dir)
    return acc, y_true, y_pred
