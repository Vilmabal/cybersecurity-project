"""Split processed feature data into train, val, and test partitions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data/processed")

def make_splits(test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    input_path = DATA_ROOT / "features.csv"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {input_path}")
    df = pd.read_csv(input_path)

    ids = df["url"].astype(str).to_numpy()
    y = df["label"].to_numpy()
    X = df.drop(columns=["label", "url"]).to_numpy(dtype=np.float32)

    #Train/test split
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(X, y, ids, test_size=test_size, stratify=y, random_state=random_state)

    #Train/validation split
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(X_train_val,y_train_val,ids_train_val,test_size=val_ratio,stratify=y_train_val,random_state=random_state)

    splits = {
        "train": (X_train, y_train, ids_train),
        "val": (X_val, y_val, ids_val),
        "test": (X_test, y_test, ids_test),
    }

    for split_name, (X_s, y_s, ids_s) in splits.items():
        np.save(DATA_ROOT / f"X_{split_name}.npy", X_s)
        np.save(DATA_ROOT / f"y_{split_name}.npy", y_s)
        np.save(DATA_ROOT / f"ids_{split_name}.npy", ids_s)
        print(f"Saved {split_name} split: {X_s.shape[0]} samples")

    #Saves feature names for interpretability/error analysis
    feature_names = df.drop(columns=["label", "url"]).columns
    pd.Series(feature_names).to_csv(DATA_ROOT / "feature_names.csv", index=False)

if __name__ == "__main__":
    make_splits()
