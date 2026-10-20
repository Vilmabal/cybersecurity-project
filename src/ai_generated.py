import pandas as pd

def print_feature_importances(importances, classifier):
    model_name = {
        "logreg": "Logistic regression",
        "random_forest": "Random forest",
    }[classifier]

    print(f"\nFeature importance: {model_name}")
    print("-" * 64)

    if classifier == "logreg":
        display = importances[
            ["feature", "weight", "absolute_weight"]
        ].copy()

        display.columns = ["Feature", "Weight", "Magnitude"]

        print(display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ))

        print("\nPositive weight: increasing the feature raises phishing probability.")
        print("Negative weight: increasing it lowers phishing probability.")
        print("These effects hold the other features fixed.")

    else:
        repeats = int(importances["n_repeats"].iloc[0])
        display = importances[
            ["feature", "mean_f1_decrease", "std_f1_decrease"]
        ].copy()
        display.columns = ["Feature", "Mean F1 drop", "SD"]

        print(f"{repeats} shuffles per feature; larger F1 drops mean greater reliance.\n")
        print(display.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        ))

        print("\nSD: variation across shuffles.")
        print("Near-zero values mean little measured effect on F1.")


def print_feature_groups(groups, classifier):
    model_name = {
        "logreg": "Logistic regression",
        "random_forest": "Random forest",
    }[classifier]
    print(f"\nFeature groups: {model_name}")
    print("-" * 64)

    binary_titles = {
        "has_ip_address": "Uses an IP address",
        "is_https": "Uses HTTPS",
        "has_punycode": "Uses Punycode",
    }

    def percentage(value):
        if pd.isna(value):
            return "N/A"
        return f"{value:.2%}"

    print("\nPerformance by feature group")
    print("Recall = phishing detected / actual phishing URLs.")
    print("False-positive rate = false alarms / actual legitimate URLs.")
    print("N/A means the metric has no valid denominator.")

    for feature, feature_rows in groups.groupby("feature", sort=False):
        title = binary_titles.get(
            feature,
            feature.replace("_", " ").capitalize(),
        )

        print(f"\n{title}")
        print("-" * 64)

        columns = {}

        for _, row in feature_rows.iterrows():
            lower_group = row["group"] == "at_or_below_threshold"

            if feature in binary_titles:
                group_label = "No" if lower_group else "Yes"
            else:
                threshold = float(row["threshold"])
                threshold_text = (
                    str(int(threshold))
                    if threshold.is_integer()
                    else str(threshold)
                )
                operator = "<=" if lower_group else ">"
                group_label = f"{operator} {threshold_text}"

            tp = int(row["true_positives"])
            fp = int(row["false_positives"])
            tn = int(row["true_negatives"])
            fn = int(row["false_negatives"])

            columns[group_label] = {
                "Test URLs": f"{int(row['num_samples']):,}",
                "Legitimate URLs": f"{int(row['num_legitimate']):,}",
                "Phishing URLs": f"{int(row['num_phishing']):,}",
                "Phishing detected (TP)": f"{tp:,}",
                "False alarms (FP)": f"{fp:,}",
                "Legitimate accepted (TN)": f"{tn:,}",
                "Phishing missed (FN)": f"{fn:,}",
                "Accuracy": percentage(row["accuracy"]),
                "Phishing precision": (
                    percentage(row["precision"]) if tp + fp else "N/A"
                ),
                "Phishing recall": (
                    percentage(row["recall"]) if tp + fn else "N/A"
                ),
                "Phishing F1": (
                    f"{row['f1']:.4f}" if 2 * tp + fp + fn else "N/A"
                ),
                "False-positive rate": (
                    percentage(row["false_positive_rate"])
                    if fp + tn else "N/A"
                ),
            }

        display = pd.DataFrame(columns)
        print(display.to_string())