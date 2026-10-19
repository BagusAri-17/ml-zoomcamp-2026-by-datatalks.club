# Notes — Module 4: Evaluation Metrics for Classification

## Course Overview
In this module, we explore **Evaluation Metrics for Classification**. In Module 3, we trained a logistic regression model to predict customer churn, achieving around 80% accuracy. However, accuracy alone is insufficient to evaluate model performance, especially under **class imbalance**. This module dives deep into comprehensive evaluation techniques for binary classifiers: accuracy and dummy baselines, the confusion table (TP, FP, FN, TN), precision, recall, ROC curves, ROC AUC (both geometric and probabilistic ranking interpretations), and **K-Fold Cross-Validation** for robust model evaluation and hyperparameter tuning.

---

## Explanation of Each Topic

### 1. Evaluation Metrics Overview
A **metric** is a function that compares model predictions with actual ground-truth targets and outputs a single quantitative number indicating how good the predictions are:
- Different metrics evaluate different dimensions of a model's behavior.
- In business contexts (like customer churn), some types of errors are far more costly than others.
- Using the Telco Customer Churn dataset, the model outputs probabilities via `predict_proba(X)[:, 1]`. By default, a decision threshold of `t = 0.5` maps probabilities to binary decisions (`churn = 1` if `prob >= 0.5`).

### 2. Accuracy and the Dummy Model
**Accuracy** measures the fraction of correct predictions over the total number of predictions:
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Threshold Optimization**: The 0.5 threshold is merely an initial guess. By sweeping thresholds `t` from 0.0 to 1.0 (e.g., using `np.linspace(0, 1, 21)`), we evaluate accuracy at each point to find the optimal decision cutoff.
- **The Dummy Model (Baseline)**: If a model predicts "no churn" for every customer (equivalent to setting threshold `t = 1.0`), it scores an accuracy of approximately **73%** simply because 73% of customers in the dataset did not churn.
- **Why Accuracy Fails (Class Imbalance)**: When one class dominates the dataset (73% negative vs. 27% positive), a naive dummy model achieves high accuracy while failing completely on the minority class—the exact customers we need to detect. Therefore, accuracy alone is misleading for imbalanced classification problems.

### 3. Confusion Table (Confusion Matrix)
The **confusion table** breaks down predictions and actual outcomes into four distinct categories to reveal the specific kinds of correct decisions and errors made by the classifier:

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | **True Negative (TN)** | **False Positive (FP)** |
| **Actual Positive (1)** | **False Negative (FN)** | **True Positive (TP)** |

- **True Positive (TP)**: Predicted churn, customer actually churned.
- **True Negative (TN)**: Predicted stayed, customer actually stayed.
- **False Positive (FP) (Type I Error)**: Predicted churn, but customer did not churn (e.g., sending unnecessary discounts or promotional emails).
- **False Negative (FN) (Type II Error)**: Predicted stayed, but customer churned (costly mistake: losing a customer without intervention).
- **Accuracy from Confusion Matrix**: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$. The table reveals the trade-offs between FP and FN that accuracy conceals.

### 4. Precision and Recall
Precision and recall focus specifically on the positive class (the minority class of interest):

- **Precision ($P$)**: The fraction of predicted positive cases that are actually correct:
  $$P = \frac{TP}{TP + FP}$$
  - *Business Context*: Out of all customers flagged for churn, what proportion actually intended to leave? Low precision means wasted promotional spending on customers who would have stayed anyway.
- **Recall ($R$) / Sensitivity / True Positive Rate (TPR)**: The fraction of actual positive cases that the model successfully caught:
  $$R = \frac{TP}{TP + FN}$$
  - *Business Context*: Out of all customers who actually left, how many did we successfully detect? Low recall means failing to prevent customer churn.
- **Mnemonics**:
  - **Pre**cision: From the **pre**dicted positives, how many are correct?
  - **Re**call: From the **re**al positives, how many did we call?
- **Trade-off & $F_1$ Score**: Increasing the threshold increases precision but decreases recall; decreasing the threshold does the opposite. The harmonic mean of precision and recall is the **$F_1$ score**:
  $$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

### 5. ROC Curves (Receiver Operating Characteristic)
Precision and recall evaluate performance at a single fixed threshold. The **ROC Curve** evaluates the classifier across **all possible decision thresholds** simultaneously by plotting **True Positive Rate (TPR)** against **False Positive Rate (FPR)**:

- **True Positive Rate (TPR)**:
  $$\text{TPR} = \frac{TP}{TP + FN} \quad (\text{Identical to Recall, want to maximize})$$
- **False Positive Rate (FPR)**:
  $$\text{FPR} = \frac{FP}{TN + FP} \quad (\text{Fraction of negatives mistakenly flagged, want to minimize})$$
- **Reference Baselines**:
  - **Random Model**: Predicts uniformly random scores. Its TPR equals FPR at every threshold, forming a diagonal line from $(0, 0)$ to $(1, 1)$.
  - **Ideal Model**: Perfectly ranks every positive above every negative. Its curve shoots vertically from $(0, 0)$ to $(0, 1)$ and then horizontally to $(1, 1)$.
- **Curve Shape**: A good model curves toward the top-left corner (high TPR with low FPR).
- **Scikit-Learn Implementation**:
  ```python
  from sklearn.metrics import roc_curve

  fpr, tpr, thresholds = roc_curve(y_val, y_pred)
  ```

### 6. ROC AUC (Area Under the ROC Curve)
The **ROC AUC** (or simply **AUC**) compresses the entire ROC curve into a single scalar value between 0.0 and 1.0:

- **Geometric Interpretation**:
  - $\text{AUC} = 0.5$: Random guessing (diagonal baseline).
  - $\text{AUC} = 1.0$: Ideal classifier.
  - *Rule of Thumb*: $0.6$ is poor, $0.7 - 0.8$ is acceptable, $0.8 - 0.9$ is good, $> 0.9$ is outstanding.
- **Probabilistic / Ranking Interpretation**:
  - AUC represents the probability that a randomly chosen positive observation receives a higher predicted score than a randomly chosen negative observation:
    $$\text{AUC} = P(\text{Score}_{\text{positive}} > \text{Score}_{\text{negative}})$$
  - This property makes AUC threshold-independent and invariant to class imbalance.
- **Scikit-Learn Implementation**:
  ```python
  from sklearn.metrics import auc, roc_auc_score

  # Via precomputed FPR and TPR
  auc(fpr, tpr)

  # Directly from true labels and predicted probabilities
  roc_auc_score(y_val, y_pred)
  ```

### 7. Cross-Validation & Parameter Tuning
Rather than relying on a single static train/validation split, **$K$-Fold Cross-Validation** provides a more reliable estimate of model performance and stability:

- **$K$-Fold Mechanism**:
  1. The full training dataset is split into $K$ equal-sized folds (partitions).
  2. The model trains on $K - 1$ folds and validates on the remaining 1 fold.
  3. The process repeats $K$ times so each fold serves as the validation set exactly once.
  4. Model performance is reported as the **mean** and **standard deviation** ($\mu \pm \sigma$) across all $K$ validation runs.
- **Hyperparameter Tuning**:
  - Used to find optimal hyperparameters (such as the regularization parameter $C$ in `LogisticRegression(C=C)`).
  - A small standard deviation indicates the model is stable across different subsets of data.
- **When to Use Cross-Validation**:
  - **Large datasets**: A single hold-out validation set is typically sufficient and computationally faster.
  - **Small to moderate datasets**: Cross-validation is recommended to avoid split variance and obtain reliable performance metrics.
- **Final Model Training**: Once the best parameter $C$ is chosen, train the final model on the entire `df_full_train` and test once on the untouched `df_test`.

---

## Summary (Module 4)
Module 4 covers essential evaluation techniques for classification systems, demonstrating why simple accuracy is misleading in the presence of class imbalance. We examined how the confusion table categorizes model outcomes into True Positives, True Negatives, False Positives, and False Negatives, and derived Precision and Recall to measure positive class performance. We explored the ROC curve across all operating thresholds, comparing against random and ideal baselines, and utilized ROC AUC as a threshold-independent measure of ranking ability. Finally, we applied $K$-Fold Cross-Validation to assess model stability and perform principled hyperparameter tuning before training the final production model.

---

## Glossary
- **Evaluation Metric**: A quantitative measure used to assess the performance of a machine learning model.
- **Class Imbalance**: A dataset condition where one target class significantly outnumbers the other class(es).
- **Dummy Baseline Model**: A naive model (e.g., always predicting the majority class) used as a baseline to benchmark true model performance.
- **Confusion Matrix**: A $2 \times 2$ table summarizing classification outcomes: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- **Type I Error (False Positive)**: Incorrectly predicting a positive outcome for an actual negative case.
- **Type II Error (False Negative)**: Incorrectly predicting a negative outcome for an actual positive case.
- **Precision**: The ratio of true positives to all predicted positives ($TP / (TP + FP)$).
- **Recall (Sensitivity / TPR)**: The ratio of true positives to all actual positives ($TP / (TP + FN)$).
- **F1 Score**: The harmonic mean of precision and recall ($2 \cdot P \cdot R / (P + R)$).
- **False Positive Rate (FPR)**: The ratio of false positives to all actual negatives ($FP / (TN + FP)$).
- **ROC Curve (Receiver Operating Characteristic)**: A graphical plot of TPR versus FPR across all possible decision thresholds.
- **ROC AUC (Area Under the ROC Curve)**: The area beneath the ROC curve; equivalent to the probability that a random positive instance is ranked higher than a random negative instance.
- **K-Fold Cross-Validation**: A resampling procedure where data is partitioned into $K$ subsets, iteratively training on $K-1$ subsets and validating on the remaining subset.
- **Regularization Parameter ($C$)**: The inverse of regularization strength in Scikit-Learn logistic regression ($C = 1 / \lambda$); smaller values specify stronger regularization.
