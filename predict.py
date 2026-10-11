"""Predicts test labels using a trained classifier and logs inference times."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

BASE_DIR = Path.cwd()
DATA_ROOT = BASE_DIR / "data" / "processed"

LEGITIMATE_LABEL = 0
PHISHING_LABEL = 1

def calculate_metrics(y_true, y_pred) -> dict:
    tn = fp = fn = tp = 0

    for i in range(len(y_true)):
        if y_true[i] == PHISHING_LABEL and y_true[i] == y_pred[i]:
            tp += 1
        elif y_true[i] == LEGITIMATE_LABEL and y_true[i] != y_pred[i]:
            fp += 1
        elif y_true[i] == PHISHING_LABEL and y_true[i] != y_pred[i]:
            fn += 1
        elif y_true[i] == LEGITIMATE_LABEL and y_true[i] == y_pred[i]:
            tn += 1
    
    return {
        "num_samples": len(y_true),
        "num_legitimate": int(tn + fp),
        "num_phishing": int(tp + fn),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=PHISHING_LABEL, zero_division=0
        ),
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
    }

def print_summary(classifier, metrics, inference_time, per_sample_time):
    model_name = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    }[classifier]

    print(f"\nTest evaluation: {model_name}")
    print("-" * 48)

    print(f"{'Number of Test URLs':<28} {metrics['num_samples']:>10,}")
    print(f"{'Number of Legitimate URLs':<28} {metrics['num_legitimate']:>10,}")
    print(f"{'Number of Phishing URLs':<28} {metrics['num_phishing']:>10,}")

    print("\nClassification outcomes")
    print(f"{'Phishing detected (TP)':<28} {metrics['true_positives']:>10,}")
    print(f"{'False alarms (FP)':<28} {metrics['false_positives']:>10,}")
    print(f"{'Legitimate accepted (TN)':<28} {metrics['true_negatives']:>10,}")
    print(f"{'Phishing missed (FN)':<28} {metrics['false_negatives']:>10,}")
    print("TP: True Positives, FP: False Positives, TN: True Negatives, FN: False Negatives")

    print("\nPerformance")
    print(f"{'Accuracy':<28} {metrics['accuracy']:>10.2%}")
    print(f"{'Phishing precision':<28} {metrics['precision']:>10.2%}")
    print(f"{'Phishing recall':<28} {metrics['recall']:>10.2%}")
    print(f"{'Phishing F1':<28} {metrics['f1']:>10.4f}")
    print(f"{'False-positive rate':<28} {metrics['false_positive_rate']:>10.2%}")

    print("\nPrediction timing")
    print(f"{'Total time':<28} {inference_time:>10.4f} s")
    print(f"{'Average time per URL':<28} {per_sample_time * 1e6:>10.2f} µs")
    print()

def predict(classifier = "random_forest", n_repeats: int = 10):

    if classifier not in {"logreg", "random_forest"}:
        raise ValueError(f"Unknown classifier: {classifier}")
    if n_repeats < 2:
        raise ValueError("Use at least two permutation repeats.")
    
    X_train = np.load(DATA_ROOT / "X_train.npy")
    X_test = np.load(DATA_ROOT / "X_test.npy")
    y_test = np.load(DATA_ROOT / "y_test.npy")
    ids_test = np.load(DATA_ROOT / "ids_test.npy", allow_pickle=True)
    feature_names = pd.read_csv(
        DATA_ROOT / "feature_names.csv"
    ).iloc[:, 0].tolist()

    if len(y_test) == 0:
        raise ValueError("Test set is empty.")
    if y_test.ndim != 1 or ids_test.ndim != 1:
        raise ValueError("Test labels and IDs must be one-dimensional.")
    if len(X_test) != len(y_test) or len(ids_test) != len(y_test):
        raise ValueError("Test features, labels, and IDs must have equal lengths.")
    if X_train.ndim != 2 or X_test.ndim != 2 or len(X_train) == 0:
        raise ValueError("Expected nonempty, two-dimensional feature matrices.")
    if X_train.shape[1] != X_test.shape[1] or len(feature_names) != X_test.shape[1]:
        raise ValueError("Feature matrices and feature names do not match.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature matrices contain missing or infinite values.")
    if not np.isin(y_test, [0, 1]).all():
        raise ValueError("Expected numeric labels 0 and 1.")

    model_path = BASE_DIR / f"{classifier}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")

    model = joblib.load(model_path)

    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Expected a model trained on both labels 0 and 1.")
    if model.n_features_in_ != len(feature_names):
        raise ValueError("Model and test data have different feature counts.")

    start = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start
    per_sample_time = inference_time / len(y_test) if len(y_test) else 0.0

    metrics = calculate_metrics(y_test, predictions)
    summary = pd.DataFrame([{
        "classifier": classifier,
        **metrics,
        "inference_time_seconds": inference_time,
        "inference_time_per_sample_seconds": per_sample_time,
    }])

    print_summary(classifier, metrics, inference_time, per_sample_time)

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

    results["outcome"] = np.select(
        [
            (y_test == 1) & (predictions == 1),
            (y_test == 0) & (predictions == 1),
            (y_test == 0) & (predictions == 0),
        ],
        ["TP", "FP", "TN"],
        default="FN",
    )

    output_path = BASE_DIR / f"predictions_{classifier}.csv"
    outputs = [
        (results, output_path),
        (summary, BASE_DIR / f"evaluation_{classifier}.csv"),
    ]

    for frame, path in outputs:
        frame.to_csv(path, index=False)
        print(f"Saved to {path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="random_forest", choices=["logreg", "random_forest"])
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of shuffles per feature for permutation importance.",
    )
    args = parser.parse_args()
    predict(classifier=args.classifier, n_repeats=args.n_repeats)
