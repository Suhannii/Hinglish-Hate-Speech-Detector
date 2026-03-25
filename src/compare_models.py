"""
compare_models.py
-----------------
Trains and evaluates multiple transformer models on the same dataset,
then plots a side-by-side comparison bar chart.

Models compared:
  - bert-base-multilingual-cased  (mBERT)
  - google/muril-base-cased       (MuRIL)

Usage:
    python src/compare_models.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.train import train_model, prepare_data
from src.evaluate import get_predictions
from src.model import get_device

MODELS = {
    "mBERT"  : "bert-base-multilingual-cased",
    "MuRIL"  : "google/muril-base-cased",
}

CSV_PATH = "data/hinglish_hate_speech.csv"
RESULTS_PATH = "models/comparison_results.json"


def evaluate_model(model_name: str, short_name: str) -> dict:
    save_dir = f"models/{short_name.lower()}_model"

    # Train
    classifier, history, test_loader = train_model(
        csv_path=CSV_PATH,
        model_name=model_name,
        epochs=3,
        batch_size=16,
        save_dir=save_dir,
    )

    # Evaluate
    device = get_device()
    y_pred, y_true = get_predictions(classifier, test_loader, device)

    return {
        "model"     : short_name,
        "model_id"  : model_name,
        "accuracy"  : round(accuracy_score(y_true, y_pred), 4),
        "f1"        : round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision" : round(precision_score(y_true, y_pred, average="macro"), 4),
        "recall"    : round(recall_score(y_true, y_pred, average="macro"), 4),
        "history"   : history,
    }


def plot_comparison(results: list, save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["accuracy", "f1", "precision", "recall"]
    labels  = [r["model"] for r in results]
    colors  = ["#a18cd1", "#f5576c", "#43e97b", "#f093fb"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for i, result in enumerate(results):
        vals = [result[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=result["model"],
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], color="white", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Model Comparison: mBERT vs MuRIL", color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("rgba(255,255,255,0.1)")
    ax.legend(facecolor="#2a2a4e", labelcolor="white", fontsize=11)
    ax.yaxis.grid(True, color="rgba(255,255,255,0.1)", linestyle="--")

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[compare] Comparison chart saved to '{path}'")


def run_comparison():
    results = []
    for short_name, model_id in MODELS.items():
        print(f"\n{'='*50}\nTraining: {short_name} ({model_id})\n{'='*50}")
        result = evaluate_model(model_id, short_name)
        results.append(result)
        print(f"  {short_name} → Acc: {result['accuracy']} | F1: {result['f1']}")

    # Save results
    os.makedirs("models", exist_ok=True)
    save_data = [{k: v for k, v in r.items() if k != "history"} for r in results]
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[compare] Results saved to '{RESULTS_PATH}'")

    # Plot
    plot_comparison(results)

    # Print summary table
    print("\n" + "=" * 50)
    print(f"{'Model':<10} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:<10} {r['accuracy']:>10.4f} {r['f1']:>8.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_comparison()
