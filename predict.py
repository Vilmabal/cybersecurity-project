"""Predict test labels using a trained classifier and log inference times."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
DATA_ROOT = PROJECT_ROOT / "data" / "processed"


def predict(classifier: str = "random_forest") -> Path:
    X_test = np.load(DATA_ROOT / "X_test.npy")
    y_test = np.load(DATA_ROOT / "y_test.npy")
    ids_test = np.load(DATA_ROOT / "ids_test.npy")

    model_path = BASE_DIR / f"{classifier}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")

    model = joblib.load(model_path)

    start = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start
    per_sample_time = inference_time / len(y_test) if len(y_test) else 0.0

    acc = accuracy_score(y_test, predictions)
    print(f"Test Accuracy ({classifier}): {acc:.4f}")
    print(f"Inference time: {inference_time:.4f}s total ({per_sample_time * 1e6:.2f} µs/sample)")

    results = pd.DataFrame(
        {
            "classifier": classifier,
            "sample_id": ids_test,
            "true_label": y_test,
            "predicted_label": predictions,
            "inference_time_seconds": inference_time,
            "inference_time_per_sample_seconds": per_sample_time,
        }
    )

    output_path = BASE_DIR / f"predictions_{classifier}.csv"
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test inference.")
    parser.add_argument(
        "--classifier",
        default="random_forest",
        choices=["logreg", "random_forest"],
        help="Trained model to use for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict(classifier=args.classifier)


if __name__ == "__main__":
    main()