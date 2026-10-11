# Homework 3: Machine Learning for Classification

📓 Full solution notebook: [homework.ipynb](./homework.ipynb)

---

## Questions & Answers

### Q1. Most frequent observation (mode) for the column industry

What is the most frequent observation (mode) for the column `industry`?

- `'finance'`
- `'healthcare'`
- `'retail'`
- **`'technology'`** ✅

**Answer: `'technology'`**

---

### Q2. Two features with the biggest correlation

What are the two features that have the biggest correlation?

- `annual_income` and `interaction_count`
- **`interaction_count` and `lead_score`** ✅
- `number_of_courses_viewed` and `interaction_count`
- `number_of_courses_viewed` and `lead_score`

**Answer: `interaction_count` and `lead_score` (correlation is 0.9287)**

---

### Q3. Feature with the biggest mutual information score

Which categorical variable has the biggest mutual information score with `converted`?

- `'employment_status'`
- `'industry'`
- **`'lead_source'`** ✅
- `'location'`

**Answer: `'lead_source'` (mutual information score is 0.03)**

---

### Q4. Model accuracy

What is the accuracy of the Logistic Regression model on the validation dataset (rounded to 2 decimal digits)?

- 0.55
- **0.65** ✅
- 0.75
- 0.85

**Answer: `0.65`**

---

### Q5. Feature with the smallest difference

Which of the following features has the smallest difference between the original accuracy and the accuracy without the feature?

- `'interaction_count'`
- `'lead_source'`
- **`'number_of_courses_viewed'`** ✅

**Answer: `'number_of_courses_viewed'` (difference is 0.002000)**

---

### Q6. Best regularization parameter C

Which value of `C` leads to the best accuracy on the validation set?

- 0.000001
- 0.00001
- 0.0001
- **0.001** ✅

**Answer: `0.001` (accuracy is 0.645)**
