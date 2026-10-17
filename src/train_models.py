"""Train URL phishing classifiers."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT/"data"/"processed"
MODEL_ROOT = PROJECT_ROOT/"models"

def build_pipeline(classifier):
    if classifier == "logreg":
        estimator = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
        return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
    elif classifier == "random_forest":
        estimator = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
        return Pipeline([("classifier", estimator)])
    else:
        raise ValueError(f"Unknown classifier, provide --logreg or keep empty for random forest classifier")

def train(classifier="random_forest"):
    print(f"Loading training data from {DATA_ROOT}")
    X_train = np.load(DATA_ROOT / "X_train.npy")
    y_train = np.load(DATA_ROOT / "y_train.npy")

    model = build_pipeline(classifier)
    print(f"Training {classifier} on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)
    
    if classifier == "logreg":
        actual_iterations = model.named_steps["classifier"].n_iter_[0]
        print(f"The model successfully converged in {actual_iterations} iterations")

    val_path = DATA_ROOT / "X_val.npy"
    if val_path.exists():
        X_val = np.load(val_path)
        y_val = np.load(DATA_ROOT / "y_val.npy")
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="binary")
        print(f"Validation Accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_ROOT / f"{classifier}_model.pkl"
    joblib.dump(model, output_path)
    print(f"Saved model: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="random_forest", choices=["logreg", "random_forest"])
    args = parser.parse_args()
    
    train(classifier=args.classifier)
