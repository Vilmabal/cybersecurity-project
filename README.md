# cybersecurity-project

## Problem statement

Phishing websites use deceptive URLs to trick people into visiting sites that may steal sensitive information. This project investigates whether machine learning can use characteristics of a URL to distinguish phishing URLs from legitimate ones.

To develop such a model, it is important to understand which features can help distinguish phishing websites from legitimate ones. Therefore, we extract different URL features and explore which of them are most useful for detecting phishing websites. We also look into two different machine learning models to determine which one is best suited for this task.

## References

1. Safi, A., & Singh, S. (2023). A systematic literature review on phishing website detection techniques. *Journal of King Saud University - Computer and Information Sciences, 35*(2), 590–611. (https://www.sciencedirect.com/science/article/pii/S1319157823000034)

2. Abu-Nimeh, S., Nappa, D., Wang, X., & Nair, S. (2007). A comparison of machine learning techniques for phishing detection. In *Proceedings of the Anti-Phishing Working Groups 2nd Annual eCrime Researchers Summit* (pp. 60–69). Association for Computing Machinery. (https://dl.acm.org/doi/abs/10.1145/1299015.1299021)

3. Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. *Expert Systems with Applications, 117*, 345–357. (https://www.sciencedirect.com/science/article/pii/S0957417418306067)

4. Abu Zuraiq, A., & Alkasassbeh, M. (2019). Review: Phishing detection approaches. In *2019 2nd International Conference on New Trends in Computing Sciences (ICTCS)*. IEEE. (https://ieeexplore.ieee.org/abstract/document/8923069)

## Documentation of project

### Project structure

```text
data/
  raw/         Original datasets
  processed/   Extracted features and train/validation/test splits
src/           Pipeline and model code
tests/         Tests
models/        Trained model files
outputs/       Predictions and evaluation reports
```

### Datasets

### Features

The project extracts URL characteristics and uses them as input for the phishing detection models. The current feature set includes URL and domain length, counts of dots, subdomains, hyphens, `@` symbols and digits, whether the URL uses HTTPS or an IP address, suspicious keywords and Punycode in the hostname, and the entropy of the URL path. These features are defined in `src/features.py` and applied to the dataset by `src/extract_features.py`.

Additional functions measure domain age, DNS records, external links and password fields. They are implemented in `src/features.py` but are not part of the current extraction list; they require network access.


### Models

### Results


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

### Alexander Granlund

I participated in group discussions and meetings throughout the project, contributing ideas about its design and implementation. My main responsibility was developing the evaluation functionality to support a fair comparison of the two classifiers.

I collaborated with the group to identify useful evaluation metrics, then researched their meaning, purpose, calculation, and interpretation. Based on this work, I implemented three functions to calculate classification metrics, assess feature importance, and evaluate classifier performance across groups of URLs with different feature characteristics. I also added input validation and error handling to the prediction function to improve the robustness of the evaluation process.

To make the results easier to interpret, I developed dedicated output functions that presented the metrics, feature importances, and feature group results in readable tables. This made it easier to identify patterns and compare performance than using the raw output alone.

Finally, I created ```compare_models.py```, a small Python script that displays selected evaluation metrics from both classifiers together to support direct comparison.

## AI usage

AI (chatGPT) was used to improve the clarity and structure of this project documentation, for example by asking: “Make this alinea clearer and more suitable for a GitHub README.”

**TODO**: report AI usage for code and put it in a folder as a seperate library!
