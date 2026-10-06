"""Split processed feature data into stratified train, val, and test partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
DATA_ROOT = PROJECT_ROOT / "data" / "processed"


def make_splits(
    input_csv: Path | str,
    output_dir: Path | str,
    target_col: str = "label",
    id_col: str = "url",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> None:
    input_path = Path(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {input_path}...")
    df = pd.read_csv(input_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_path}")

    # Preserve URL or unique identifiers if present
    if id_col in df.columns:
        ids = df[id_col].astype(str).to_numpy()
        features_df = df.drop(columns=[target_col, id_col])
    else:
        ids = np.asarray([f"url_{i:06d}" for i in range(len(df))])
        features_df = df.drop(columns=[target_col])

    X = features_df.to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy()

    # Train / Test split
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X, y, ids, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train / Validation split
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=random_state,
    )

    splits = {
        "train": (X_train, y_train, ids_train),
        "val": (X_val, y_val, ids_val),
        "test": (X_test, y_test, ids_test),
    }

    for split_name, (X_s, y_s, ids_s) in splits.items():
        np.save(out_path / f"X_{split_name}.npy", X_s)
        np.save(out_path / f"y_{split_name}.npy", y_s)
        np.save(out_path / f"ids_{split_name}.npy", ids_s)
        print(f"Saved {split_name} split: {X_s.shape[0]} samples (features={X_s.shape[1]})")

    # Save feature names for interpretability / error analysis
    pd.Series(features_df.columns).to_csv(out_path / "feature_names.csv", index=False)
    print(f"Feature names written to {out_path / 'feature_names.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DATA_ROOT / "features.csv",
        help="Path to feature matrix CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT,
        help="Directory to save split .npy files.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="label",
        help="Name of target label column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_splits(input_csv=args.input_csv, output_dir=args.output_dir, target_col=args.target_col)


if __name__ == "__main__":
    main()