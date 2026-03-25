"""
main.py
-------
Entry point for the entire pipeline.
Run this file to execute all steps end-to-end:
  Step 1 → Generate dataset
  Step 2 → Train model
  Step 3 → Evaluate model + save plots
  Step 4 → Run demo predictions

Usage:
    python main.py
"""

import os
import sys
import pandas as pd

# ── Make sure src/ is importable ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Configuration (edit these if needed) ──────────────────────────────────────
CONFIG = {
    "dataset_path"  : "data/hinglish_hate_speech.csv",
    "model_name"    : "google/muril-base-cased",  # change to "ai4bharat/indic-bert" etc.
    "epochs"        : 3,
    "batch_size"    : 16,
    "learning_rate" : 2e-5,
    "max_length"    : 128,
    "save_dir"      : "models/muril_model",
    "plots_dir"     : "plots",
}


def banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── Step 1: Generate Dataset ───────────────────────────────────────────────────
def step1_generate_dataset():
    banner("STEP 1: Generating Synthetic Hinglish Dataset")
    from src.dataset_generator import generate_dataset
    df = generate_dataset(output_path=CONFIG["dataset_path"], augment=True)
    return df


# ── Step 2: Train Model ────────────────────────────────────────────────────────
def step2_train():
    banner("STEP 2: Training Transformer Model")
    from src.train import train_model
    classifier, history, test_loader = train_model(
        csv_path       = CONFIG["dataset_path"],
        model_name     = CONFIG["model_name"],
        epochs         = CONFIG["epochs"],
        batch_size     = CONFIG["batch_size"],
        learning_rate  = CONFIG["learning_rate"],
        max_length     = CONFIG["max_length"],
        save_dir       = CONFIG["save_dir"],
    )
    return classifier, history, test_loader


# ── Step 3: Evaluate ───────────────────────────────────────────────────────────
def step3_evaluate(classifier, history, test_loader, df):
    banner("STEP 3: Evaluating Model")
    from src.evaluate import run_evaluation, plot_label_distribution
    plot_label_distribution(df, save_dir=CONFIG["plots_dir"])
    acc, y_true, y_pred = run_evaluation(
        classifier, test_loader, history, save_dir=CONFIG["plots_dir"]
    )
    return acc


# ── Step 4: Demo Predictions ───────────────────────────────────────────────────
def step4_predict():
    banner("STEP 4: Running Demo Predictions")
    from src.predict import HateSpeechPredictor, demo_predictions
    predictor = HateSpeechPredictor(
        model_dir  = CONFIG["save_dir"],
        base_model = CONFIG["model_name"],
        max_length = CONFIG["max_length"],
    )
    demo_predictions(predictor)

    # Interactive loop
    print("\n[Interactive Mode] Type a Hinglish sentence to classify.")
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting. Goodbye!")
            break
        if not user_input:
            continue
        result = predictor.predict(user_input)
        print(f"  → Prediction : {result['label']}")
        print(f"  → Confidence : {result['confidence']}%")
        print(f"  → Probs      : Non-Hate={result['probabilities']['Non-Hate']}%  "
              f"Hate={result['probabilities']['Hate']}%\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  Hinglish Hate Speech Detection — Full Pipeline")
    print("#" * 60)

    df            = step1_generate_dataset()
    classifier, history, test_loader = step2_train()
    step3_evaluate(classifier, history, test_loader, df)
    step4_predict()
