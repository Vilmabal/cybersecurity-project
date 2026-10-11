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
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

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

    output_path = Path(f"predictions_{classifier}.csv")
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="random_forest", choices=["logreg", "random_forest"])
    args = parser.parse_args()
    
    predict(classifier=args.classifier)
