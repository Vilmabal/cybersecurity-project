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

Run the pipeline from the repository root with `python -m src/main.py`.

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

### Lena Geentjens

I focused mainly on identifying and extracting useful features from URLs for our phishing detection models. I began by reading research papers to learn more about the topic and identify features that could help a machine learning model distinguish phishing websites from legitimate ones. I also explored which models would be suitable for our project.

Then, in `features.py`, I implemented functions to extract these features. Some were straightforward, such as URL length and the number of dots, while others were more complex and required network requests and socket connections. These functions were used to transform URLs into data that the models could process.

After training the models, I participated in discussions about the results and helped interpret them, including identifying which features appeared to be the most influential. I also helped improve the project’s documentation and reorganize its directory structure, making the project easier to understand and navigate.
