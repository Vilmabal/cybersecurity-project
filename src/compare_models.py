import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

comparison = pd.concat(
    [
        pd.read_csv(OUTPUT_ROOT / "evaluation_logreg.csv"),
        pd.read_csv(OUTPUT_ROOT / "evaluation_random_forest.csv"),
    ],
    ignore_index=True,
)

print(comparison[[
    "classifier",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
]].to_string(index=False))
