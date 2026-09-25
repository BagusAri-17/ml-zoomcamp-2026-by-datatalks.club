# Notes — Module 1: Introduction to Machine Learning

## Course Overview
In this course, I learned the fundamentals of machine learning, the types of algorithms involved, how to determine the best model, and the environments used for machine learning projects (such as the frequently employed packages and libraries).

---

## Explanation of Each Topic

### 1. What is Machine Learning
**Machine Learning** is a process of extracting patterns and insights from data.
In ML, I studied 2 types of ML data:
- **Features**: Information about the data or object. Example: the age and characteristics of a car.
- **Target**: What we want to predict. Example: the price of a car.

In ML, there is also the term **Model Training**. Training here means providing features and targets to a **Machine Learning Algorithm** to create a model. Example: a model can predict the price of a car based on its features.
In conclusion, the output of the machine learning process is a model that takes information about an object and processes it based on learned patterns to generate automated predictions, decisions, or classifications.

### 2. Machine Learning vs Rules
Example: A spam detection problem.
In a **Rules-based approach**, we extract rules ourselves and write them in code. The data (email) and the code together form software, and the software produces outcomes (detects spam or not spam).
In **Machine Learning**, the outcomes become the input for a machine learning algorithm, alongside the data. The algorithm then produces a model for prediction.

ML can solve problems (e.g., spam detection) using the following steps:
1. Get Data: Collect data, such as emails.
2. Define & Calculate Features: The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox). Each email can be encoded (converted) into the values of its features and target.
3. Train & Use Model: A machine learning algorithm can then be applied to the encoded emails to build a model that predicts whether a new email is spam or not.

### 3. Supervised Machine Learning
**Supervised Machine Learning (SML)** is about teaching an algorithm by showing it examples. 
The examples go into the:
- Feature matrix **X**: all the characteristics of the objects we want to make predictions for.
- Target vector **y**: the target we want to predict.

We pass X and y into a machine learning algorithm and train the model. The model is usually denoted as `g`: it is a function that takes the feature matrix X as input and produces an output that is approximately close to the target y.

**g(X) ≈ y**

The **goal** of SML is to find this function `g` such that when we apply it to X, the output is as close as possible to the target variable.

There are several types of SML problems:
- **Regression**: The output can be any continuous number. Any problem where the output is a number is a regression problem. Example: predicting car prices or the number of rooms.
- **Classification**: The output is a category. Example: spam detection (spam or not spam). Classification has subclasses:
    - **Binary Classification**: The output has exactly two categories. The target is 0 or 1, and `g` outputs a probability between 0 and 1. Example: email spam detection.
    - **Multiclass Classification**: The output has more than two categories. Example: classifying images into cats, dogs, and birds.
- **Ranking**: Used in recommender systems. It involves a function that scores every item, sorts them by score, and shows the top results. Example: an e-commerce website showing a list of recommended products.

### 4. CRISP-DM
**CRISP-DM**, which stands for **Cross-Industry Standard Process for Data Mining**, is an open standard process model that describes common approaches used by data mining experts.
For any ML project, we need to understand the problem, collect the data, train the model, and use it. CRISP-DM helps us organize these steps in a manageable way, so we know what needs to happen and in what order.
The CRISP-DM process has six steps:
- Step 1 - Business Understanding: Identify the problem we want to solve.
- Step 2 - Data Understanding: Understand what data is available, whether it is sufficient, and how we can obtain what is missing (e.g., buying a data source or collecting data ourselves).
- Step 3 - Data Preparation: Transform data so it can be fed into a machine learning algorithm. Transformation involves extracting features, cleaning data, removing noise, building data pipelines, and converting data into a tabular format.
- Step 4 - Modeling: Start training the model. This is where the actual machine learning happens. We try different models and select the best one.
- Step 5 - Evaluation: Measure how well the best model performs.
- Step 6 - Deployment: Roll out the model to production for users.

Don't forget to constantly iterate on the process, learn from feedback, and improve.

### 5. Model Selection
The **Multiple Comparisons Problem (MCP)** occurs when one model appears to obtain good predictions purely by chance because models are probabilistic.
A test set can help avoid the MCP. Selecting the best model is done using the training and validation datasets, while the test dataset is used to confirm that the chosen model truly is the best.

1. Split datasets into training, validation, and test sets. E.g., 60%, 20%, and 20% respectively.
2. Train the models.
3. Evaluate the models.
4. Select the best model.
5. Apply the best model to the test dataset.
6. Compare the performance metrics of the validation and test sets.

*Note: It is possible to reuse the validation data. After selecting the best model (step 4), the validation and training datasets can be combined to form a single training dataset to retrain the chosen model before finally testing it on the test set.*

### 6. Environment
We need to set up the environment before starting the course. This includes Python, Jupyter Notebook, and libraries or packages useful in this course, such as NumPy, Pandas, Scikit-Learn, Matplotlib, and Seaborn.
For notebook services, we can use Kaggle, Google Colab, or a local setup.

### 7. Numpy
**NumPy**, short for Numerical Python, provides functions for creating arrays, multi-dimensional arrays, randomly generated arrays, element-wise operations, comparison operations, and summarizing operations.

### 8. Linear Algebra
**Linear Algebra** covers simple vector operations, the three kinds of multiplication (vector-vector, matrix-vector, and matrix-matrix), and two special objects - the identity matrix and the inverse matrix.

### 9. Pandas
**Pandas** is a library for manipulating tabular data in Python. We use it to work with DataFrames and Series, including indexing and accessing elements, element-wise operations, filtering, string operations, summarizing operations, handling missing values, and grouping.

---

## Summary
Module 1 introduces the foundational concepts of Machine Learning, explaining the difference between traditional rule-based programming and ML. It covers Supervised Machine Learning techniques (Regression, Classification, and Ranking), model selection strategies to prevent overfitting (like the Multiple Comparisons Problem), and outlines the standard ML project lifecycle using CRISP-DM. It also touches on essential tools and libraries (NumPy, Pandas, Linear Algebra) needed for data manipulation and modeling.

---

## Glossary
- **Machine Learning**: The process of extracting patterns and insights from data to build predictive models.
- **Features (X)**: The input variables or characteristics of the data used for making predictions.
- **Target (y)**: The output variable that the model is trying to predict.
- **Supervised Machine Learning**: Training an algorithm using labeled examples (features and their corresponding targets).
- **Regression**: A supervised learning task where the target output is a continuous numerical value.
- **Classification**: A supervised learning task where the target output is a categorical label.
- **Ranking**: A machine learning task focused on scoring and sorting items, commonly used in recommender systems.
- **CRISP-DM**: A standard six-step process for organizing data mining and ML projects (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment).
- **Multiple Comparisons Problem**: The statistical phenomenon where evaluating many models might yield a seemingly \"best\" model purely by chance.
- **NumPy**: A Python library specialized for numerical computations and handling multi-dimensional arrays.
- **Pandas**: A Python library used for data manipulation and analysis, primarily through DataFrames and Series.
