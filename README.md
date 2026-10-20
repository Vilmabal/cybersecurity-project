# cybersecurity-project

## Problem statement

Phishing websites use deceptive URLs to trick people into visiting sites that may steal sensitive information. This project investigates whether machine learning can use characteristics of a URL to distinguish phishing URLs from legitimate ones.

To develop such a model, it is important to understand which features can help distinguish phishing websites from legitimate ones. Therefore, we extract different URL features and explore which of them are most useful for detecting phishing websites. We also look into two different machine learning models to determine which one is best suited for this task.

## References

1. Safi, A., & Singh, S. (2023). A systematic literature review on phishing website detection techniques. *Journal of King Saud University - Computer and Information Sciences, 35*(2), 590–611. (https://www.sciencedirect.com/science/article/pii/S1319157823000034)

2. Abu-Nimeh, S., Nappa, D., Wang, X., & Nair, S. (2007). A comparison of machine learning techniques for phishing detection. In *Proceedings of the Anti-Phishing Working Groups 2nd Annual eCrime Researchers Summit* (pp. 60–69). Association for Computing Machinery. (https://dl.acm.org/doi/abs/10.1145/1299015.1299021)

3. Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. *Expert Systems with Applications, 117*, 345–357. (https://www.sciencedirect.com/science/article/pii/S0957417418306067)

4. Abu Zuraiq, A., & Alkasassbeh, M. (2019). Review: Phishing detection approaches. In *2019 2nd International Conference on New Trends in Computing Sciences (ICTCS)*. IEEE. (https://ieeexplore.ieee.org/abstract/document/8923069)

5. Kaitholikkal, J. K. S., & Arthi, B. (2024). Phishing URL dataset (Version 1). (https://www.phishtank.com/developer_info.php)

6. Cisco Talos Intelligence Group. (2026). PhishTank Developer Information: Verified Online Phish Feed. (https://data.mendeley.com/datasets/vfszbj9b36/1)

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

The dataset used in this project contains a balanced distribution of 153,822 total URLs, split equally between malicious and benign. Each URL is labelled either 0 if legitimate or 1 if phishing. The malicious URLs are extracted directly from the PhishTank dataset, and the benign URLs are extracted from the Phishing URL dataset (Kaitholikkal & Arthi, 2024). 

Because the source benign dataset contained a much larger number of 345,738 legitimate entries, a random probability filter was applied, with p=0.226, via a Rainbow Query Language (RBQL) expression (`Math.random() < 0.226 LIMIT 76911`). This extracted roughly 22.6% of the given database, before hitting the capping threshold.

To construct the final unified dataset, both components were merged using an Excel spreadsheet. The stacked data blocks were completely randomized and shuffled using a dummy calculation field populated by the =RAND() function, sorting the data frames from smallest to largest value, and subsequently discarding the generator column.

### Features

The project extracts URL characteristics and uses them as input for the phishing detection models. The current feature set includes URL and domain length, counts of dots, subdomains, hyphens, `@` symbols and digits, whether the URL uses HTTPS or an IP address, suspicious keywords and Punycode in the hostname, and the entropy of the URL path. These features are defined in `src/features.py` and applied to the dataset by `src/extract_features.py`.

Additional functions measure domain age, DNS records, external links and password fields. They are implemented in `src/features.py` but are not part of the current extraction list; they require network access.


### Models

2 machine learning algorithms were evaluated to compare a linear baseline against a non-linear model:

1. Logistic regression: This serves as a more interpretable baseline to evaluate linear separability of the extracted URL features and is built using "StandardScaler" to normalize features prior to model fitting since it's essential because of our different scaling features (e.g. URL length vs binary flags). It is oprimized using the lbfgs solver with L2 regularization (C=1). To ensure the optimization algorithm fully converges, the maximum iterations were increased from the default 100 to 2000 and convergence iterations are logged during training. 

2. Random forest classifier: An ensemble of 100 decision trees designed to capture non-linear relationships among URL features. Unlike logistic regression, it does not need feature normalization. The model is trained with unrestricted tree depth (max_depth=None) for each tree to learn detailed patterns. 

For the data splitting, stratified partitioning was used to ensure identical ratios of legitimate and phishing URL's across all subsets which consisted of: Train (70%), validation (10%) and test (20%) sets. Feature matrices, labels, URL identifiers and column names are saved to `data/processed/` for evaluation and error inspection. Both models are trained with a fixed random seed for reproducibility and random forest runs in parallel (`n_jobs=-1`) to accelerate training. After fitting on the training set, intermediate validation performance is also monitored, using accuracy and F1-score to verify convergence. The models are then saved to `models/*.pkl` using `joblib`.

### Results



### predict.py

This function calculates the performance metrics: Number of Test URLs, Number of Legitimate URLs, Number of Phishing URLs, True Positives, False Positives, True Negatives, False Negatives, accuracy, precision, recall and F1. 

```
def calculate_metrics(y_true, y_pred) -> dict:
```

This function quantifies and calculates the feature importance for the currently running model. By feature importance we mean how this feature influences the model, or how much the models performance depends on it.
```
def calculate_feature_importance(model, classifier, X_test, y_test, feature_names, n_repeats) -> pd.DataFrame:
```

This function quantifies and calculates what effect feature groups has on the URLs. By this we mean how accurately the model classify URLs with this certain feature group. 
```
def calculate_feature_groups(X_train, X_test, y_test, predictions, feature_names) -> pd.DataFrame:
```

This function prints a summary of the metrics provided by calculate_metrics and the additional metrics inference time and per sample time. 
```
def print_summary(classifier, metrics, inference_time, per_sample_time):
```

This function prints a summary of the metrics provided by calcualte_feature_importance.
```
def print_feature_importances(importances, classifier):
```

This function prints a summary of the metrics provided by calcualte_feature_groups.
```
def print_feature_groups(groups, classifier):
```

###Data from running **Random Forest** classifier (with default value of 10 repeats for feature importance)

Test evaluation: Random forest
------------------------------------------------
```
Number of Test URLs              30,765
Number of Legitimate URLs        15,383
Number of Phishing URLs          15,382

Classification outcomes
Phishing detected (TP)           14,210
False alarms (FP)                   906
Legitimate accepted (TN)         14,477
Phishing missed (FN)              1,172
```
TP: True Positives, FP: False Positives, TN: True Negatives, FN: False Negatives

Performance
```
Accuracy                         93.25%
Phishing precision               94.01%
Phishing recall                  92.38%
Phishing F1                      0.9319
False-positive rate               5.89%

Prediction timing
Total time                       0.0848 s
Average time per URL               2.76 µs
```
Feature importance: Random forest
----------------------------------------------------------------
10 shuffles per feature; larger F1 drops mean greater reliance.
```
                 Feature       Mean F1 drop      SD
           domain_length       0.17935           0.00154
    number_of_subdomains       0.09162           0.00095
        url_path_entropy       0.07189           0.00125
              url_length       0.06725           0.00124
        number_of_digits       0.05201           0.00105
          number_of_dots       0.05165           0.00172
       number_of_hyphens       0.02438           0.00097
                is_https       0.02028           0.00070
suspicious_keyword_count       0.00267           0.00028
    number_of_at_symbols       0.00161           0.00018
          has_ip_address       0.00000           0.00000
            has_punycode      -0.00001           0.00002
```
SD: variation across shuffles.
Near-zero values mean little measured effect on F1.

Feature groups: Random forest
----------------------------------------------------------------

Performance by feature group
Recall = phishing detected / actual phishing URLs.
False-positive rate = false alarms / actual legitimate URLs.
N/A means the metric has no valid denominator.

Url length
----------------------------------------------------------------
```
                           <= 46    > 46
Test URLs                 15,849  14,916
Legitimate URLs            5,566   9,817
Phishing URLs             10,283   5,099
Phishing detected (TP)     9,508   4,702
False alarms (FP)            757     149
Legitimate accepted (TN)   4,809   9,668
Phishing missed (FN)         775     397
Accuracy                  90.33%  96.34%
Phishing precision        92.63%  96.93%
Phishing recall           92.46%  92.21%
Phishing F1               0.9254  0.9451
False-positive rate       13.60%   1.52%
```

Domain length
----------------------------------------------------------------
```
                           <= 19    > 19
Test URLs                 16,113  14,652
Legitimate URLs            8,405   6,978
Phishing URLs              7,708   7,674
Phishing detected (TP)     7,094   7,116
False alarms (FP)            260     646
Legitimate accepted (TN)   8,145   6,332
Phishing missed (FN)         614     558
Accuracy                  94.58%  91.78%
Phishing precision        96.46%  91.68%
Phishing recall           92.03%  92.73%
Phishing F1               0.9420  0.9220
False-positive rate        3.09%   9.26%
```
Number of dots
----------------------------------------------------------------
```
                            <= 2     > 2
Test URLs                 20,318  10,447
Legitimate URLs            7,353   8,030
Phishing URLs             12,965   2,417
Phishing detected (TP)    12,058   2,152
False alarms (FP)            829      77
Legitimate accepted (TN)   6,524   7,953
Phishing missed (FN)         907     265
Accuracy                  91.46%  96.73%
Phishing precision        93.57%  96.55%
Phishing recall           93.00%  89.04%
Phishing F1               0.9328  0.9264
False-positive rate       11.27%   0.96%
```
Number of subdomains
----------------------------------------------------------------
```
                            <= 1     > 1
Test URLs                 26,301   4,464
Legitimate URLs           11,736   3,647
Phishing URLs             14,565     817
Phishing detected (TP)    13,477     733
False alarms (FP)            887      19
Legitimate accepted (TN)  10,849   3,628
Phishing missed (FN)       1,088      84
Accuracy                  92.49%  97.69%
Phishing precision        93.82%  97.47%
Phishing recall           92.53%  89.72%
Phishing F1               0.9317  0.9344
False-positive rate        7.56%   0.52%
```
Number of hyphens
----------------------------------------------------------------
```
                            <= 0     > 0
Test URLs                 17,082  13,683
Legitimate URLs            9,183   6,200
Phishing URLs              7,899   7,483
Phishing detected (TP)     7,017   7,193
False alarms (FP)            697     209
Legitimate accepted (TN)   8,486   5,991
Phishing missed (FN)         882     290
Accuracy                  90.76%  96.35%
Phishing precision        90.96%  97.18%
Phishing recall           88.83%  96.12%
Phishing F1               0.8989  0.9665
False-positive rate        7.59%   3.37%
```
Number of at symbols
----------------------------------------------------------------
```
                            <= 0      > 0
Test URLs                 30,622      143
Legitimate URLs           15,367       16
Phishing URLs             15,255      127
Phishing detected (TP)    14,083      127
False alarms (FP)            902        4
Legitimate accepted (TN)  14,465       12
Phishing missed (FN)       1,172        0
Accuracy                  93.23%   97.20%
Phishing precision        93.98%   96.95%
Phishing recall           92.32%  100.00%
Phishing F1               0.9314   0.9845
False-positive rate        5.87%   25.00%
```
Number of digits
----------------------------------------------------------------
```
                            <= 1     > 1
Test URLs                 16,240  14,525
Legitimate URLs            8,244   7,139
Phishing URLs              7,996   7,386
Phishing detected (TP)     7,113   7,097
False alarms (FP)            761     145
Legitimate accepted (TN)   7,483   6,994
Phishing missed (FN)         883     289
Accuracy                  89.88%  97.01%
Phishing precision        90.34%  98.00%
Phishing recall           88.96%  96.09%
Phishing F1               0.8964  0.9703
False-positive rate        9.23%   2.03%
```
Uses an IP address
----------------------------------------------------------------
```
                              No      Yes
Test URLs                 30,744       21
Legitimate URLs           15,383        0
Phishing URLs             15,361       21
Phishing detected (TP)    14,189       21
False alarms (FP)            906        0
Legitimate accepted (TN)  14,477        0
Phishing missed (FN)       1,172        0
Accuracy                  93.24%  100.00%
Phishing precision        94.00%  100.00%
Phishing recall           92.37%  100.00%
Phishing F1               0.9318   1.0000
False-positive rate        5.89%      N/A
```
Uses HTTPS
----------------------------------------------------------------
```
                               No     Yes
Test URLs                   1,362  29,403
Legitimate URLs                 0  15,383
Phishing URLs               1,362  14,020
Phishing detected (TP)      1,362  12,848
False alarms (FP)               0     906
Legitimate accepted (TN)        0  14,477
Phishing missed (FN)            0   1,172
Accuracy                  100.00%  92.93%
Phishing precision        100.00%  93.41%
Phishing recall           100.00%  91.64%
Phishing F1                1.0000  0.9252
False-positive rate           N/A   5.89%
```

Suspicious keyword count
----------------------------------------------------------------
```
                            <= 0     > 0
Test URLs                 30,403     362
Legitimate URLs           15,353      30
Phishing URLs             15,050     332
Phishing detected (TP)    13,884     326
False alarms (FP)            902       4
Legitimate accepted (TN)  14,451      26
Phishing missed (FN)       1,166       6
Accuracy                  93.20%  97.24%
Phishing precision        93.90%  98.79%
Phishing recall           92.25%  98.19%
Phishing F1               0.9307  0.9849
False-positive rate        5.88%  13.33%
```
Uses Punycode
----------------------------------------------------------------
```
                              No      Yes
Test URLs                 30,746       19
Legitimate URLs           15,383        0
Phishing URLs             15,363       19
Phishing detected (TP)    14,192       18
False alarms (FP)            906        0
Legitimate accepted (TN)  14,477        0
Phishing missed (FN)       1,171        1
Accuracy                  93.24%   94.74%
Phishing precision        94.00%  100.00%
Phishing recall           92.38%   94.74%
Phishing F1               0.9318   0.9730
False-positive rate        5.89%      N/A
```
Url path entropy
----------------------------------------------------------------
```
                         <= 2.9219281673431396 > 2.9219281673431396
Test URLs                               15,362               15,403
Legitimate URLs                          4,228               11,155
Phishing URLs                           11,134                4,248
Phishing detected (TP)                  10,438                3,772
False alarms (FP)                          740                  166
Legitimate accepted (TN)                 3,488               10,989
Phishing missed (FN)                       696                  476
Accuracy                                90.65%               95.83%
Phishing precision                      93.38%               95.78%
Phishing recall                         93.75%               88.79%
Phishing F1                             0.9356               0.9216
False-positive rate                     17.50%                1.49%
```

###Data from running **Logistic Regression** classifier

Test evaluation: Logistic regression
------------------------------------------------
```
Number of Test URLs              30,765
Number of Legitimate URLs        15,383
Number of Phishing URLs          15,382

Classification outcomes
Phishing detected (TP)           12,346
False alarms (FP)                 2,204
Legitimate accepted (TN)         13,179
Phishing missed (FN)              3,036
```
TP: True Positives, FP: False Positives, TN: True Negatives, FN: False Negatives

Performance
```
Accuracy                         82.97%
Phishing precision               84.85%
Phishing recall                  80.26%
Phishing F1                      0.8249
False-positive rate              14.33%

Prediction timing
Total time                       0.0033 s
Average time per URL               0.11 µs
```
Feature importance: Logistic regression
----------------------------------------------------------------
```
                 Feature  Weight  Magnitude
                is_https -2.2448     2.2448
        number_of_digits  2.0348     2.0348
              url_length  1.5352     1.5352
    number_of_subdomains -1.4712     1.4712
          number_of_dots -0.8893     0.8893
        url_path_entropy -0.8250     0.8250
           domain_length  0.6547     0.6547
       number_of_hyphens -0.5060     0.5060
    number_of_at_symbols  0.4817     0.4817
suspicious_keyword_count  0.3895     0.3895
            has_punycode  0.2423     0.2423
          has_ip_address  0.0926     0.0926
```
Positive weight: increasing the feature raises phishing probability.
Negative weight: increasing it lowers phishing probability.
These effects hold the other features fixed.

Feature groups: Logistic regression
----------------------------------------------------------------

Performance by feature group
Recall = phishing detected / actual phishing URLs.
False-positive rate = false alarms / actual legitimate URLs.
N/A means the metric has no valid denominator.

Url length
----------------------------------------------------------------
```
                           <= 46    > 46
Test URLs                 15,849  14,916
Legitimate URLs            5,566   9,817
Phishing URLs             10,283   5,099
Phishing detected (TP)     8,591   3,755
False alarms (FP)          1,593     611
Legitimate accepted (TN)   3,973   9,206
Phishing missed (FN)       1,692   1,344
Accuracy                  79.27%  86.89%
Phishing precision        84.36%  86.01%
Phishing recall           83.55%  73.64%
Phishing F1               0.8395  0.7934
False-positive rate       28.62%   6.22%
```
Domain length
----------------------------------------------------------------
```
                           <= 19    > 19
Test URLs                 16,113  14,652
Legitimate URLs            8,405   6,978
Phishing URLs              7,708   7,674
Phishing detected (TP)     5,938   6,408
False alarms (FP)          1,028   1,176
Legitimate accepted (TN)   7,377   5,802
Phishing missed (FN)       1,770   1,266
Accuracy                  82.64%  83.33%
Phishing precision        85.24%  84.49%
Phishing recall           77.04%  83.50%
Phishing F1               0.8093  0.8400
False-positive rate       12.23%  16.85%
```
Number of dots
----------------------------------------------------------------
```
                            <= 2     > 2
Test URLs                 20,318  10,447
Legitimate URLs            7,353   8,030
Phishing URLs             12,965   2,417
Phishing detected (TP)    11,010   1,336
False alarms (FP)          2,012     192
Legitimate accepted (TN)   5,341   7,838
Phishing missed (FN)       1,955   1,081
Accuracy                  80.48%  87.81%
Phishing precision        84.55%  87.43%
Phishing recall           84.92%  55.28%
Phishing F1               0.8473  0.6773
False-positive rate       27.36%   2.39%
```
Number of subdomains
----------------------------------------------------------------
```
                            <= 1     > 1
Test URLs                 26,301   4,464
Legitimate URLs           11,736   3,647
Phishing URLs             14,565     817
Phishing detected (TP)    12,019     327
False alarms (FP)          2,193      11
Legitimate accepted (TN)   9,543   3,636
Phishing missed (FN)       2,546     490
Accuracy                  81.98%  88.78%
Phishing precision        84.57%  96.75%
Phishing recall           82.52%  40.02%
Phishing F1               0.8353  0.5662
False-positive rate       18.69%   0.30%
```
Number of hyphens
----------------------------------------------------------------
```
                            <= 0     > 0
Test URLs                 17,082  13,683
Legitimate URLs            9,183   6,200
Phishing URLs              7,899   7,483
Phishing detected (TP)     6,181   6,165
False alarms (FP)          1,829     375
Legitimate accepted (TN)   7,354   5,825
Phishing missed (FN)       1,718   1,318
Accuracy                  79.24%  87.63%
Phishing precision        77.17%  94.27%
Phishing recall           78.25%  82.39%
Phishing F1               0.7770  0.8793
False-positive rate       19.92%   6.05%
```
Number of at symbols
----------------------------------------------------------------
```
                            <= 0      > 0
Test URLs                 30,622      143
Legitimate URLs           15,367       16
Phishing URLs             15,255      127
Phishing detected (TP)    12,231      115
False alarms (FP)          2,188       16
Legitimate accepted (TN)  13,179        0
Phishing missed (FN)       3,024       12
Accuracy                  82.98%   80.42%
Phishing precision        84.83%   87.79%
Phishing recall           80.18%   90.55%
Phishing F1               0.8244   0.8915
False-positive rate       14.24%  100.00%
```
Number of digits
----------------------------------------------------------------
```
                            <= 1     > 1
Test URLs                 16,240  14,525
Legitimate URLs            8,244   7,139
Phishing URLs              7,996   7,386
Phishing detected (TP)     6,191   6,155
False alarms (FP)          1,453     751
Legitimate accepted (TN)   6,791   6,388
Phishing missed (FN)       1,805   1,231
Accuracy                  79.94%  86.35%
Phishing precision        80.99%  89.13%
Phishing recall           77.43%  83.33%
Phishing F1               0.7917  0.8613
False-positive rate       17.62%  10.52%
```
Uses an IP address
----------------------------------------------------------------
```
                              No      Yes
Test URLs                 30,744       21
Legitimate URLs           15,383        0
Phishing URLs             15,361       21
Phishing detected (TP)    12,325       21
False alarms (FP)          2,204        0
Legitimate accepted (TN)  13,179        0
Phishing missed (FN)       3,036        0
Accuracy                  82.96%  100.00%
Phishing precision        84.83%  100.00%
Phishing recall           80.24%  100.00%
Phishing F1               0.8247   1.0000
False-positive rate       14.33%      N/A
```
Uses HTTPS
----------------------------------------------------------------
```
                               No     Yes
Test URLs                   1,362  29,403
Legitimate URLs                 0  15,383
Phishing URLs               1,362  14,020
Phishing detected (TP)      1,362  10,984
False alarms (FP)               0   2,204
Legitimate accepted (TN)        0  13,179
Phishing missed (FN)            0   3,036
Accuracy                  100.00%  82.18%
Phishing precision        100.00%  83.29%
Phishing recall           100.00%  78.35%
Phishing F1                1.0000  0.8074
False-positive rate           N/A  14.33%
```
Suspicious keyword count
----------------------------------------------------------------
```
                            <= 0     > 0
Test URLs                 30,403     362
Legitimate URLs           15,353      30
Phishing URLs             15,050     332
Phishing detected (TP)    12,019     327
False alarms (FP)          2,196       8
Legitimate accepted (TN)  13,157      22
Phishing missed (FN)       3,031       5
Accuracy                  82.81%  96.41%
Phishing precision        84.55%  97.61%
Phishing recall           79.86%  98.49%
Phishing F1               0.8214  0.9805
False-positive rate       14.30%  26.67%
```
Uses Punycode
----------------------------------------------------------------
```
                              No      Yes
Test URLs                 30,746       19
Legitimate URLs           15,383        0
Phishing URLs             15,363       19
Phishing detected (TP)    12,327       19
False alarms (FP)          2,204        0
Legitimate accepted (TN)  13,179        0
Phishing missed (FN)       3,036        0
Accuracy                  82.96%  100.00%
Phishing precision        84.83%  100.00%
Phishing recall           80.24%  100.00%
Phishing F1               0.8247   1.0000
False-positive rate       14.33%      N/A
```
Url path entropy
----------------------------------------------------------------
```
                         <= 2.9219281673431396 > 2.9219281673431396
Test URLs                               15,362               15,403
Legitimate URLs                          4,228               11,155
Phishing URLs                           11,134                4,248
Phishing detected (TP)                   9,551                2,795
False alarms (FP)                        1,738                  466
Legitimate accepted (TN)                 2,490               10,689
Phishing missed (FN)                     1,583                1,453
Accuracy                                78.38%               87.54%
Phishing precision                      84.60%               85.71%
Phishing recall                         85.78%               65.80%
Phishing F1                             0.8519               0.7444
False-positive rate                     41.11%                4.18%
```

Summary of raw data between the two classifiers:
```
classifier               true_positives    false_positives  false_negatives  precision   recall       f1          false_positive_rate
logreg                   12346             2204             3036             0.848522    0.802626     0.824937    0.143275
random_forest            14210             906              1172             0.940064    0.923807     0.931864    0.058896
```
We can from this data and these metrics determine that the Random Forest classifier is the superior model (on this data) with an approximate 10 percentage point difference in precision, recall and f1 score. 

## Documentation on testing the project

To try out and run the project only the main.py program has to be run. Only one classifier can be run at a time. When the main.py program runs the collected URLs are split into three groups: training, validation and testing. Then the program trains the chosen classifier on the training data and validates with the validation data. The program then predicts the nature of the URLs in the testing data (phishing or legitimate) and after these predictions have been made the evaluation functions are run to calculate and display the performance of the chosen classifier. 

The program should be run from the root project directory to not encounter any program path logic errors. Therefore use src/file.py.
```
python3 src/main.py
```
Runs the program on default values, that is classifier = random forest and number of repeats = 10.

To specify model, use the --classifier <classifier> argument. (To choose logistic regression, use arg "logreg"

```
python3 src/main.py --classifier logreg
```

To specify number of repeats for the feature importance shuffle when running the random forest classifier, use the --n-repeats <n-repeats> argument. 

```
python src/main.py --classifier random_forest n-repeats 30
```
This then changes the number of repeats (each feature shuffle) from the default 10 to 30.


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

### Vilma Balicevac

My responsibilities were implementing the data splits, the model training and the main file which runs the whole pipeline. I created the partitioning of the processed feature data into training, validation and testing and applied stratified sampling across them. I also ensured the URL identifiers and feature names were preserved to support analysis later on.  

I also implemented the training pipeline for the classifiers. For logistic regression, I constructed a scikit-learn `Pipeline` integrating `StandardScaler` to ensure numerical stability and proper convergence with the lbfgs solver. For random forest, I implemented an ensemble of 100 trees. I added automated validation checks for monitoring the model fit for both models. 

To bring all stages together, I implemented `src/main.py` as the end-to-end command-line runner where i added argument parsing  (`--classifier`) to easily switch between models when running, input caching to avoid redundant feature extraction, and error handling. 

### Cecilia Leotta

My primary responsibility for this project focused on data acquisition and dataset standardization to ensure a balanced, binary-labeled input for the classifiers. While the phishing URLs were extracted directly from the PhishTank database, sourcing legitimate URLs required utilizing the Phishing URL Dataset (Kaitholikkal & Arthi, 2024). Due to the structural complexity of mapping safe internet traffic, a public repository of benign URLs similar to PhishTank doesn’t exist. The Phishing URL dataset is also used in multiple peer reviewed studies, which makes it the optimal choice for this project.

To handle the significantly larger size of the benign dataset, I applied a random probability filter to maintain a proportional and unbiased sample size. I then merged the two distinct data streams and randomized the final sequence using a spreadsheet-based calculation field populated by the =RAND() function to eliminate any positional bias.

Finally, I utilized Excel to design and generate the descriptive statistical visualizations and data distribution graphs for our final presentation slides.


## AI usage

AI (chatGPT) was used to improve the clarity and structure of this project documentation, for example by asking: “Make this alinea clearer and more suitable for a GitHub README.”

AI was used for for checking and questions regarding code syntax. 

AI was used for code debugging.

AI was used for and summaries of research and the interpretation of certain performance metrics. 

AI was used to make the the two functions: print_feature_importances and print_feature_groups since these functions only purpose was to structure data in a readable way. 


**TODO**: report AI usage for code and put it in a folder as a seperate library!
