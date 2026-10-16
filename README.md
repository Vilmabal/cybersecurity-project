# cybersecurity-project

## Project structure

```text
data/
  raw/         Original datasets
  processed/   Extracted features and train/validation/test splits
src/           Pipeline and model code
tests/         Tests
models/        Trained model files
outputs/       Predictions and evaluation reports
```

Run the pipeline from the repository root with `python -m src.main`.

## Problem statement

## Reference

## Documentation of project

### predict.py

This function....

```
def calculate_metrics(y_true, y_pred) -> dict:
```

```
def calculate_feature_importance(model, classifier, X_test, y_test, feature_names, n_repeats) -> pd.DataFrame:
```

```
def calculate_feature_groups(X_train, X_test, y_test, predictions, feature_names) -> pd.DataFrame:
```

```
def print_summary(classifier, metrics, inference_time, per_sample_time):
```

```
def print_feature_importances(importances, classifier):
```

```
def print_feature_groups(groups, classifier):
```

## Documentation on testing the project

## Group members contributions
