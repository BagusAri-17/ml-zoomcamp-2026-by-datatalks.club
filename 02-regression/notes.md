# Notes — Module 2: Machine Learning for Regression

## Course Overview
In this module, we focus on **Machine Learning for Regression**. The primary goal is to predict a continuous numerical value (e.g., the price of a car) based on various features. We build a complete end-to-end ML project, starting from data preparation, exploratory data analysis (EDA), setting up a robust validation framework, implementing linear regression from scratch, and finally evaluating the model's performance.

---

## Explanation of Each Topic

### 1. Car Price Intro
The objective of this project is to build a machine learning model to predict car prices based on their characteristics.

**The project plan includes:**
- Downloading and preparing the dataset.
- Performing Exploratory Data Analysis (EDA) to understand data distributions.
- Setting up a validation framework to evaluate model performance reliably.
- Implementing Linear Regression to predict the target variable (price).
- Understanding the internal mathematics of Linear Regression.
- Evaluating the model using the RMSE metric.
- Improving the model via Feature Engineering and Regularization.

**Dataset link:** [Kaggle Car Features and MSRP](https://www.kaggle.com/CooperUnion/cardataset)

### 2. Data Preparation
Data preparation involves cleaning and formatting the raw data so it can be consumed by machine learning algorithms. We use the **Pandas** library for data manipulation.

**Key Pandas Commands:**
- `pd.read_csv(<file_path>)`: Read CSV files into a DataFrame.
- `df.head()`: View the first 5 rows of the DataFrame.
- `df.columns`: Retrieve the column names.
- `df.columns.str.lower()`: Lowercase all column names for consistency.
- `df.columns.str.replace(' ', '_')`: Replace spaces with underscores in column names.
- `df.dtypes`: Retrieve data types of all features (useful for separating numerical and categorical features).
- `df.index`: Retrieve indices of a DataFrame.

### 3. EDA (Exploratory Data Analysis)
EDA helps us understand the characteristics of our data, find missing values, and analyze the distribution of our target variable (`price`).

**Pandas Commands for EDA:**
- `df[col].unique()`: Return a list of unique values in a specific column.
- `df[col].nunique()`: Return the number of unique values.
- `df.isnull().sum()`: Return the number of missing (null) values per column.

**Data Visualization (Matplotlib & Seaborn):**
- `%matplotlib inline`: Ensures that plots are displayed directly in Jupyter Notebook cells.
- `sns.histplot(df[col])`: Shows the histogram of a variable to visualize its distribution.

**Handling Long-Tail Distributions:**
- Often, price data has a "long tail" (many cheap cars, few extremely expensive ones), which can confuse our ML model.
- We apply a logarithmic transformation to compress this tail and make the distribution closer to a normal distribution.
- `np.log1p(df['price'])`: Applies log transformation after adding 1 to each value (to avoid `log(0)` errors).

### 4. Validation Framework
To evaluate our model properly, we split the dataset into three distinct partitions:
1. **Training set (~60%)**: Used to train the model.
2. **Validation set (~20%)**: Used to evaluate the model during training and tune hyperparameters.
3. **Test set (~20%)**: Used for the final evaluation of the model after everything is finalized.

**Steps to Split the Data:**
1. Determine the size (number of rows) for each partition based on the percentages.
2. Shuffle the dataset indices using a fixed random seed (for reproducibility) to ensure data is randomly distributed across sets.
3. Use the shuffled indices to split the data.
4. Separate the target variable (`y`) from the feature matrices (`X`) for all three sets.

**Useful Commands:**
- `np.random.seed(2)`: Sets the seed for reproducibility.
- `np.random.permutation(n)` or `np.random.shuffle(idx)`: Shuffles an array of indices.
- `df.iloc[indices]`: Subsets records of a DataFrame using numerical indices.
- `df.reset_index(drop=True)`: Resets the indices of the resulting DataFrames so they start from 0.
- `del df['price']`: Removes the target column from the feature matrix so the model doesn't cheat.

### 5. Linear Regression (Simple)
Linear Regression is a fundamental model where the objective is to fit a line (or hyperplane) to the data that best maps the input features to the target values. 

**Formula:**
`y = w0 + w1*x1 + w2*x2 + ... + wn*xn`
- `y`: The predicted target value.
- `x1, x2, ...`: The input features of a single record.
- `w0`: The bias (or y-intercept). It represents the base prediction when all features are 0.
- `w1, w2, ...`: The weights. They represent how much the target variable increases/decreases when the corresponding feature increases by 1 unit.

The model aims to find the optimal values for the weights (`w`) that minimize the distance between the predictions and the actual `y` values.

### 6. Linear Regression (Vector Form)
Linear regression can be represented in vector form to compute predictions efficiently for all observations at once. We use the dot product of the feature matrix `X` and the weights vector `w`. 
- `X.dot(w)` gives the predictions `y_pred`.
- Bias term `w0` can be incorporated into `w` by adding a dummy column of ones to `X`.

### 7. Linear Regression Training (Normal Equation)
To find the optimal weights `w` that minimize the error between predictions and actual targets, we use the Normal Equation:
`w = (X^T X)^-1 X^T y`
In NumPy:
- `XTX = X.T.dot(X)`
- `XTX_inv = np.linalg.inv(XTX)`
- `w_full = XTX_inv.dot(X.T).dot(y)`

### 8. Baseline Model
A baseline model is a simple initial model used to establish a reference point for performance. In the car price project, the baseline model is a simple linear regression using a basic set of numerical features (like engine HP, year, doors, etc.) without complex preprocessing or feature engineering.

### 9. RMSE (Root Mean Squared Error)
RMSE is the standard metric for evaluating regression models. It measures the average magnitude of the errors between predictions and actual values.
Formula: `RMSE = sqrt( (1/m) * sum( (y_pred - y)^2 ) )`
- Lower RMSE indicates a better model.

### 10. Validating the Model
The RMSE is calculated on the training data and validation data. If the model is good, the RMSE on both datasets should be similar. If the training RMSE is much lower than the validation RMSE, the model is overfitting.

### 11. Feature Engineering
Creating new features from existing ones to improve model performance. For example, calculating the `age` of a car using the `year` feature (`age = 2017 - year`).

### 12. Categorical Variables
Machine learning models require numerical input. Categorical variables (like make, model, transmission type) must be converted into numerical format, typically using **One-Hot Encoding**. 
- In Pandas, this can be done manually or via `pd.get_dummies()`.
- We create new binary features for each unique category value.

### 13. Regularization
Regularization helps prevent overfitting by penalizing large weights in the model. We add a small number (alpha) to the diagonal of the `X^T X` matrix before inverting it:
`XTX = X.T.dot(X) + alpha * np.eye(XTX.shape[0])`
This prevents matrix invertibility issues (singular matrix) and keeps weights smaller and more generalized.

### 14. Tuning the Model
Finding the best hyperparameter (like the `alpha` value in regularization) by training models with different alpha values and comparing their RMSE on the **validation dataset**. The alpha that yields the lowest validation RMSE is selected.

### 15. Using the Model
Once the best model and hyperparameters are found, the model is trained one final time on the combined **training + validation** dataset. Then, the final performance is evaluated on the unseen **test dataset**. If the test RMSE is consistent with the validation RMSE, the model is ready to be used on new data.

---

## Summary (Module 2)
Module 2 focuses on building a complete machine learning pipeline for a regression task (predicting car prices). It covers the entire process from data preparation, EDA, and setting up a validation framework, to understanding the math behind Linear Regression (Normal Equation) and evaluating the model using RMSE. Advanced concepts like feature engineering, one-hot encoding for categorical variables, and regularization to prevent overfitting are also introduced to improve model robustness.

---

## Glossary
- **Linear Regression**: A model that assumes a linear relationship between the input variables (features) and the single output variable (target).
- **Weights (`w`)**: The coefficients learned by the model that determine the importance of each feature.
- **Normal Equation**: An analytical solution to find the optimal weights for Linear Regression.
- **RMSE**: Root Mean Squared Error, a metric to measure the difference between predicted and actual values in regression.
- **Feature Engineering**: The process of using domain knowledge to create new features that make ML algorithms work better.
- **One-Hot Encoding**: A process of converting categorical variables into a numerical form (binary vectors).
- **Regularization**: A technique used to reduce overfitting by adding a penalty term to the model's complexity (e.g., Ridge Regression penalizes large weights).
- **Baseline Model**: A simple model used as a reference point to compare against more complex models.
