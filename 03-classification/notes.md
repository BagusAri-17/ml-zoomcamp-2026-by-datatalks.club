# Notes — Module 3: Machine Learning for Classification

## Course Overview
In this module, we focus on **Machine Learning for Classification**. Unlike regression where we predict a continuous number, classification is about predicting a categorical outcome (e.g., yes/no, spam/not spam). We build an end-to-end classification project focused on **customer churn prediction**, covering data preparation, exploratory data analysis, measuring feature importance (using risk ratio, mutual information, and correlation), and finally implementing and interpreting **Logistic Regression** using Scikit-Learn.

---

## Explanation of Each Topic

### 1. Churn Prediction Project
The objective of this project is to build a machine learning model to predict **customer churn** (whether a customer will stop using a service). This is a **binary classification** problem where the target variable has two possible values: `0` (no churn) and `1` (churn).

**Dataset link:** Telco Customer Churn dataset from Kaggle.

### 2. Data Preparation
Just like in regression, data preparation involves cleaning and formatting the data:
- Standardizing column names and string values (lowercasing and replacing spaces with underscores).
- Converting columns to the correct data types (e.g., converting a `TotalCharges` column from string to numeric).
- Handling missing values (e.g., replacing nulls with `0`).

### 3. Setting Up the Validation Framework
Using `scikit-learn` to split the dataset instead of doing it manually with NumPy:
- `train_test_split` from `sklearn.model_selection` is used to divide the data into training, validation, and test sets (typically in a 60/20/20 ratio).
- Target variables (`y`) are extracted and removed from the feature matrices (`X`).

### 4. EDA (Exploratory Data Analysis)
Analyzing the distribution of the target variable (churn rate) and exploring the characteristics of both numerical and categorical features.
- Checking for missing values.
- Understanding the global churn rate (the mean of the target variable).

### 5. Feature Importance: Churn Rate and Risk Ratio
Understanding which categorical variables are most important for predicting churn by comparing group metrics against global metrics.
- **Group Churn Rate**: The churn rate within a specific category (e.g., churn rate for females vs. males).
- **Risk Ratio**: The ratio of the group churn rate to the global churn rate. 
  - `Risk > 1`: The group is more likely to churn (high risk).
  - `Risk < 1`: The group is less likely to churn (low risk).

### 6. Feature Importance: Mutual Information
**Mutual Information (MI)** is a concept from information theory that measures how much information we learn about one variable by knowing another. 
- It is used to measure the importance of **categorical features**. 
- Higher MI means the feature is more informative for predicting the target.
- In Scikit-Learn: `mutual_info_score` from `sklearn.metrics`.

### 7. Feature Importance: Correlation
To measure the importance of **numerical features**, we use the **Pearson correlation coefficient** ($r$).
- Measures the linear dependency between two variables (from -1 to 1).
- Positive correlation: As the feature increases, the target (churn probability) increases.
- Negative correlation: As the feature increases, the target decreases.

### 8. One-Hot Encoding
Converting categorical variables into numerical format so they can be fed into machine learning models.
- Using Scikit-Learn's `DictVectorizer`:
  - Converts dictionaries into vectors.
  - Automatically applies one-hot encoding to categorical variables while leaving numerical variables untouched.

### 9. Logistic Regression
Despite its name, Logistic Regression is a **classification algorithm**. 
- It works similarly to linear regression but passes the output through a **Sigmoid function** to map any real-valued number into a probability between 0 and 1.
- **Formula:** $g(x_i) = Sigmoid(w_0 + w^T x_i)$
- **Sigmoid function:** $\sigma(z) = \frac{1}{1 + e^{-z}}$

### 10. Training Logistic Regression with Scikit-Learn
Using `LogisticRegression` from `sklearn.linear_model` to train the model on our one-hot encoded training data.
- The `fit(X, y)` method trains the model.
- The `predict_proba(X)` method outputs a 2D array with probabilities for each class (e.g., probability of 0 and probability of 1). We typically take the second column (`[:, 1]`) as the probability of churn.

### 11. Model Interpretation
Understanding how the model makes decisions by looking at the learned coefficients (weights `w`) and the bias (`w0`).
- A positive weight increases the overall probability (increases churn risk).
- A negative weight decreases the probability (decreases churn risk).
- We can manually multiply a single customer's features by the weights to see exactly how their specific characteristics contribute to their final churn probability.

### 12. Using the Model
Combining the training and validation datasets to train a final model. Then, testing this final model on the untouched test set to ensure the accuracy holds up. We can then use this trained model to predict the probability of churn for new customers.

---

## Summary (Module 3)
Module 3 teaches how to solve binary classification problems. Using customer churn as the primary example, we covered data preparation, using `scikit-learn` for data splitting and model training, and evaluating feature importance using Risk Ratio, Mutual Information, and Correlation. We learned how to use `DictVectorizer` for one-hot encoding and finally trained, interpreted, and used a Logistic Regression model to predict the probability of a categorical outcome.

---

## Glossary
- **Binary Classification**: A machine learning task where the output belongs to one of two mutually exclusive classes (e.g., 0 or 1, Yes or No).
- **Churn Rate**: The percentage of customers who stop using a company's product or service.
- **Risk Ratio**: The ratio of the risk of an event occurring in an exposed group to the risk of the event in an unexposed (or global) group.
- **Mutual Information**: A metric from information theory that measures the mutual dependence between two variables (best for categorical features).
- **Correlation**: A statistical measure that indicates the extent to which two variables fluctuate together (best for numerical features).
- **DictVectorizer**: A Scikit-Learn utility that transforms lists of feature-value mappings (dictionaries) into vectors, applying one-hot encoding to categorical features.
- **Sigmoid Function**: A mathematical function having a characteristic "S"-shaped curve, used to map predictions to probabilities between 0 and 1.
- **Logistic Regression**: A statistical model used for binary classification that uses the sigmoid function to output probabilities.
