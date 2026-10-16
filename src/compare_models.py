import pandas as pd

comparison = pd.concat(
    [
        pd.read_csv("outputs/evaluation_logreg.csv"),
        pd.read_csv("outputs/evaluation_random_forest.csv"),
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
