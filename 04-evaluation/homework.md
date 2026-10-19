# Homework 4: Evaluation Metrics for Classification

📓 Full solution notebook: [homework.ipynb](./homework.ipynb)

---

## Questions & Answers

### Q1. ROC AUC feature importance

Which numerical variable (among the following 4) has the highest AUC?

- **`lead_score`** ✅
- `number_of_courses_viewed`
- `interaction_count`
- `annual_income`

**Answer: `lead_score` (AUC is 0.7882)**

---

### Q2. Training the model

What's the AUC of this model on the validation dataset? (round to 3 digits)

- 0.532
- 0.632
- **0.732** ✅
- 0.832

**Answer: `0.732` (Validation AUC is 0.7318)**

---

### Q3. Precision and Recall

At which threshold precision and recall curves intersect? Ignore thresholds at which both metrics are zero, and choose the threshold with the smallest absolute difference between precision and recall.

- 0.43
- **0.63** ✅
- 0.73
- 0.83

**Answer: `0.63` (Precision is 0.7328, Recall is 0.7253, difference is 0.0075)**

---

### Q4. F1 score

At which threshold F1 is maximal?

- 0.21
- **0.41** ✅
- 0.61
- 0.81

**Answer: `0.41` (Maximal F1 score is 0.7576)**

---

### Q5. 5-Fold CV

How large is standard deviation of the scores across different folds?

- 0.001
- **0.007** ✅
- 0.013
- 0.060

**Answer: `0.007` (standard deviation is 0.0069)**

---

### Q6. Hyperparameter Tuning

Which `C` leads to the best mean ROC AUC?

- 0.000001
- **0.001** ✅
- 1

**Answer: `0.001` (mean ROC AUC is 0.749, std is 0.010)**
