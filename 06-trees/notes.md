# Notes — Module 6: Decision Trees and Ensemble Learning

## Course Overview
In this module, we explore **Decision Trees and Ensemble Learning**, transitioning from linear and generalized linear models (e.g., Logistic Regression) to non-linear tree-based algorithms. Tree-based models are among the most robust, versatile, and widely used machine learning methods for tabular structured data in competitive data science and industry.

Throughout this module, we tackle a real-world **Credit Risk Scoring** project: predicting whether a bank customer will default on a loan. We begin with the foundation of decision trees—how they split data using if-then-else rules, how tree learning algorithms optimize split criteria (Gini Impurity, Entropy, MSE), and why unconstrained trees overfit. We then advance to **Ensemble Learning**, exploring **Random Forest** (parallel bagging with feature subsampling) and **Gradient Boosting** via **XGBoost** (sequential learning on residual errors). Finally, we systematically tune hyperparameters and evaluate the winning model on unseen test data.

---

## Explanation of Each Topic

### 1. Credit Risk Scoring Project
Credit scoring is a foundational use case in financial services. When an individual applies for a loan or credit card, the lender must assess the risk of default:
- **Business Goal**: Maximize approved profitable loans while minimizing financial loss from defaults.
- **Classification Problem**:
  - `default` ($y = 1$): High risk — the client fails to repay the loan.
  - `ok` ($y = 0$): Low risk — the client successfully repays the loan.
- **Dataset Context**: The CreditScoring dataset contains client financial attributes including employment history, income, loan amount, home ownership, marital status, assets, debt, and historical credit records.
- **Why Tree Models?**: Tabular datasets in banking often exhibit complex non-linear relationships, multi-level categorical features, and varying feature scales where linear models require extensive manual feature transformation, whereas tree-based ensembles handle them naturally.

### 2. Data Cleaning and Preparation
Real-world data rarely arrives ready for machine learning. The CreditScoring dataset requires preprocessing:

#### Standardizing Column Names
Column names are converted to lowercase to ensure consistency:
```python
df.columns = df.columns.str.lower()
```

#### Decoding Categorical Features
In many legacy databases, categorical features are stored as arbitrary integer codes. We remap them back to descriptive strings:
```python
status_values = {1: 'ok', 2: 'default', 0: 'unk'}
df.status = df.status.map(status_values)

home_values = {
    1: 'rent', 2: 'owner', 3: 'private', 4: 'ignore', 5: 'parents', 6: 'other', 0: 'unk'
}
df.home = df.home.map(home_values)

marital_values = {1: 'single', 2: 'married', 3: 'widow', 4: 'separated', 5: 'divorced', 0: 'unk'}
df.marital = df.marital.map(marital_values)

records_values = {1: 'no', 2: 'yes', 0: 'unk'}
df.records = df.records.map(records_values)

job_values = {1: 'fixed', 2: 'parttime', 3: 'freelance', 4: 'others', 0: 'unk'}
df.job = df.job.map(job_values)
```

#### Handling Sentinel Missing Values
Numeric columns encode missing values using sentinel numbers (specifically `99999999`). Leaving these raw values would distort statistics and numerical splits:
```python
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)
```

#### Removing Unknown Target Rows
A single row in the dataset contains `status == 'unk'`. Because we cannot train or validate a supervised model on unknown targets, this record is removed:
```python
df = df[df.status != 'unk'].reset_index(drop=True)
```

#### Train / Validation / Test Splitting
We split the data into 60% training, 20% validation, and 20% testing using Scikit-Learn's `train_test_split` with `random_state=11`:
```python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'default').astype(int).values
y_val = (df_val.status == 'default').astype(int).values
y_test = (df_test.status == 'default').astype(int).values

del df_train['status']
del df_val['status']
del df_test['status']
```

#### Feature Vectorization
We use `DictVectorizer(sparse=False)` to convert records into numeric matrices:
```python
from sklearn.feature_extraction import DictVectorizer

train_dicts = df_train.fillna(0).to_dict(orient='records')
val_dicts = df_val.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
```

---

### 3. Decision Trees & The Overfitting Problem

#### What is a Decision Tree?
A Decision Tree is a non-parametric supervised learning model that partitions feature space into distinct rectangular regions using hierarchical if-then-else conditions:
- **Root Node**: The top node where the first split occurs.
- **Internal Nodes**: Decision junctions testing a feature condition (e.g., `assets > 3000`).
- **Branches**: Outcomes of the decision test (`True` or `False`).
- **Leaf (Terminal) Nodes**: End nodes containing final predictions (class labels or probabilities).

```
                 [ records == yes? ]
                     /         \
                 Yes /           \ No
                    /             \
             [ assets > 3000? ]   [ seniority > 2? ]
               /          \          /          \
             Yes          No       Yes          No
          [ P=0.21 ]   [ P=0.68 ] [ P=0.10 ]  [ P=0.35 ]
```

#### Decision Stump
A **decision stump** is the simplest possible tree, having a maximum depth of 1 (`max_depth=1`). It makes a single binary split on the most informative feature:
```python
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(max_depth=1)
dt.fit(X_train, y_train)

print(export_text(dt, feature_names=list(dv.get_feature_names_out())))
```

#### The Overfitting Problem in Decision Trees
When trained without constraints (`max_depth=None`, `min_samples_leaf=1`), a decision tree continues splitting until every leaf contains only a single sample or all samples belong to the same class:
- **Training Set Performance**: $AUC = 1.000$ (or near 100% accuracy). The model memorizes individual noise and outliers.
- **Validation Set Performance**: $AUC \approx 0.654$. The tree fails to generalize to unseen data.

> [!WARNING]
> Unconstrained decision trees will almost always overfit. They create overly deep, complex partition boundaries that capture idiosyncratic quirks of the training sample rather than generalizable data patterns.

To prevent overfitting, we apply regularization via stopping criteria:
- `max_depth`: Limits how deep the tree can grow.
- `min_samples_leaf`: Enforces a minimum number of samples required to form a leaf node.

---

### 4. Decision Tree Learning Algorithm (Splitting & Impurity)

#### Recursive Binary Splitting
Decision tree algorithms (such as CART — Classification and Regression Trees) use a greedy, top-down approach:
1. Iterate over every feature $j$.
2. For numeric features, sort unique values and evaluate midpoints as candidate split thresholds $T$.
3. For categorical features, evaluate binary membership conditions.
4. Select the feature $j^*$ and threshold $T^*$ that maximizes the reduction in impurity (or information gain).
5. Split the node into left ($X_j \le T$) and right ($X_j > T$) subsets.
6. Recursively repeat the procedure on child nodes until a stopping condition is met.

#### Measuring Impurity for Classification
Impurity measures the degree of class mixture within a node:

1. **Misclassification Error Rate**:
   $$E(m) = 1 - \max_k (p_k)$$
   where $p_k$ is the proportion of samples belonging to class $k$ in node $m$. While intuitive, it is not sensitive enough to changes in node probabilities to build optimal trees.

2. **Gini Impurity (CART default)**:
   $$Gini(m) = 1 - \sum_{k=1}^K p_k^2$$
   For binary classification ($p$ = positive class proportion):
   $$Gini(m) = 1 - [p^2 + (1 - p)^2] = 2p(1 - p)$$
   - Pure node ($p=0$ or $p=1$): $Gini = 0.0$ (minimum impurity).
   - Maximum mixture ($p=0.5$): $Gini = 0.5$ (maximum impurity).

3. **Entropy / Information Gain (ID3, C4.5)**:
   $$Entropy(m) = -\sum_{k=1}^K p_k \log_2(p_k)$$
   For binary classification:
   $$Entropy(m) = -p \log_2(p) - (1-p) \log_2(1-p)$$
   - Pure node: $Entropy = 0.0$.
   - Maximum mixture ($p=0.5$): $Entropy = 1.0$.

#### Weighted Impurity Reduction
A split divides a parent node containing $N$ samples into left ($N_L$ samples) and right ($N_R$ samples) children. The split quality is evaluated by the weighted average impurity:
$$I_{split} = \frac{N_L}{N} I(L) + \frac{N_R}{N} I(R)$$
The best split minimizes $I_{split}$ (or maximizes Information Gain: $\Delta I = I(Parent) - I_{split}$).

#### Regression Trees
When predicting continuous targets:
- Node prediction is the mean target value of samples in that node: $\hat{y} = \frac{1}{N} \sum_{i=1}^N y_i$.
- Impurity is measured via **Mean Squared Error (MSE)** / Variance:
  $$MSE(m) = \frac{1}{N} \sum_{i \in m} (y_i - \hat{y}_m)^2$$

---

### 5. Parameter Tuning for Decision Trees
To maximize generalization, we tune two key hyperparameters jointly:
- `max_depth`: Tree capacity ($1, 2, 3, 4, 5, 6, 10, 15, 20$).
- `min_samples_leaf`: Minimum node size ($1, 2, 5, 10, 15, 20, 100$).

#### Parameter Grid Search
```python
from sklearn.metrics import roc_auc_score

scores = []

for depth in [1, 2, 3, 4, 5, 6, 10, 15, 20]:
    for s in [1, 2, 5, 10, 15, 20, 100]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s, random_state=1)
        dt.fit(X_train, y_train)
        
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((depth, s, auc))

df_scores = pd.DataFrame(scores, columns=['max_depth', 'min_samples_leaf', 'auc'])
```

#### Visualizing with Heatmaps
A 2D pivot table visualized with Seaborn makes optimal hyperparameter regions immediately clear:
```python
import seaborn as sns
import matplotlib.pyplot as plt

df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns='max_depth', values='auc')

plt.figure(figsize=(10, 6))
sns.heatmap(df_scores_pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('Validation AUC across max_depth and min_samples_leaf')
plt.show()
```

#### Optimal Configuration:
- `max_depth = 6` with `min_samples_leaf = 15` achieves the highest validation AUC ($\approx 0.785$).
- Shallow trees (`max_depth <= 2`) underfit (AUC $\approx 0.70$).
- Deep unconstrained trees (`max_depth >= 15`, `min_samples_leaf = 1`) overfit (AUC $\approx 0.65$).

---

### 6. Ensemble Learning & Random Forest

#### The Principle of Ensemble Learning
Ensemble learning combines predictions from multiple individual models ("weak learners") to create a single superior predictor ("strong learner"). This embodies the **Wisdom of the Crowds**: while individual models make mistakes, their uncorrelated errors cancel out when aggregated.

For ensembling to succeed:
1. **Model Diversity**: Individual models must make different errors.
2. **Better Than Random Guessing**: Each model must perform better than pure chance ($AUC > 0.5$).

#### Bagging (Bootstrap Aggregating)
Bagging creates diversity by training identical algorithms on different subsets of training data:
- **Bootstrapping**: Given a dataset of size $N$, draw $N$ samples uniformly **with replacement**. On average, each bootstrap sample contains $\approx 63.2\%$ unique original rows; the remaining $\approx 36.8\%$ are omitted (Out-Of-Bag samples).
- **Aggregation**: Average predictions across all models (for regression or probabilities) or take the majority vote (for classification).

#### Random Forest
A Random Forest extends bagging to decision trees with an extra source of diversity:
- **Bootstrap Sampling**: Each tree is trained on a distinct bootstrap sample.
- **Random Feature Subsampling**: At *every split node*, the algorithm selects a random subset of features (typically $\sqrt{p}$ for classification, where $p$ is the total number of features). Only features within this subset are evaluated for the split.

```
                  Training Data (N samples, P features)
                    /             |              \
           Bootstrap 1       Bootstrap 2     Bootstrap M
                |                 |               |
             Tree 1            Tree 2          Tree M
         (subsampled feat) (subsampled feat) (subsampled feat)
                \                 |               /
                 ------> Average Probabilities <------
                                  |
                           Final Prediction
```

> [!NOTE]
> Why feature subsampling? Without it, a single dominant feature (e.g., `records == yes`) would be chosen at the root of almost every tree, producing highly correlated trees. Restricting candidate features forces trees to explore alternative predictive signals.

#### Scikit-Learn Implementation & Hyperparameters
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=1,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
print("Random Forest AUC:", roc_auc_score(y_val, y_pred)) # ~0.825
```

#### Key Hyperparameters:
- `n_estimators`: Number of trees. As $n\_estimators$ increases, performance improves and reaches an asymptotic plateau. **Adding more trees does not cause overfitting in Random Forest.**
- `max_depth`: Maximum depth of each individual tree (typically 10–15).
- `min_samples_leaf`: Minimum samples per leaf (e.g., 3–5).
- `n_jobs=-1`: Utilizes all CPU cores for parallel tree training.

---

### 7. Gradient Boosting & XGBoost

#### Bagging vs. Boosting
| Attribute | Bagging (Random Forest) | Boosting (Gradient Boosting) |
| :--- | :--- | :--- |
| **Training Process** | Independent, parallel | Sequential, iterative |
| **Model Role** | Deep, diverse trees | Shallow, specialized weak learners |
| **Core Mechanism** | Averages independent predictions | Successive trees predict residual errors |
| **Risk of Overfitting** | High number of trees does NOT overfit | Too many trees WILL overfit |
| **Tuning Complexity** | Low (works well out of the box) | High (requires careful hyperparameter tuning) |

#### How Gradient Boosting Works
Gradient boosting trains models sequentially. Each new tree $h_m(x)$ is trained to predict the negative gradient (residual errors) of the loss function made by the current ensemble $F_{m-1}(x)$:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where:
- $F_0(x)$: Initial baseline prediction (e.g., the log-odds of the positive class).
- $h_m(x)$: New decision tree trained on the pseudo-residuals $r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F=F_{m-1}}$.
- $\eta$ (`eta`): Learning rate (shrinkage factor), scaling each tree's contribution (e.g., $0.1$ or $0.05$).

```
 Iteration 0: Baseline Model F_0(x)  -----> Residuals e_0 = y - F_0(x)
                                                  |
 Iteration 1: Train Tree h_1(x) on e_0 ----------> F_1(x) = F_0(x) + η * h_1(x)
                                                  |
                                             Residuals e_1 = y - F_1(x)
                                                  |
 Iteration 2: Train Tree h_2(x) on e_1 ----------> F_2(x) = F_1(x) + η * h_2(x)
```

#### Introduction to XGBoost
**XGBoost (Extreme Gradient Boosting)** is an optimized, highly scalable library implementing gradient boosted decision trees. It includes second-order Taylor expansion approximations, tree pruning, native sparsity awareness, column subsampling, and GPU acceleration.

#### Using `xgb.DMatrix`
XGBoost uses its own internal optimized data structure called `DMatrix`:
```python
import xgboost as xgb

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
```

#### Training and Monitoring Learning Curves
We pass a `watchlist` to monitor performance on both train and validation sets across boosting rounds:
```python
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dval, 'val')]
evals_result = {}

model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=200,
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=False
)
```

---

### 8. XGBoost Hyperparameter Tuning
Because XGBoost can easily overfit with inappropriate settings, tuning must be executed in a structured, sequential order:

```
 Step 1: Learning Rate (eta)  ──►  Step 2: Tree Depth (max_depth)  ──►  Step 3: Leaf Weight (min_child_weight)  ──►  Step 4: Optimal Rounds
```

#### Step 1: Tune `eta` (Learning Rate)
`eta` controls how fast the model learns:
- Large `eta` (e.g., $0.3$): Fast convergence, but risks oscillating and early overfitting.
- Moderate `eta` (e.g., $0.1$): Good balance between convergence speed and peak generalization.
- Small `eta` (e.g., $0.01$): Slower convergence, requires hundreds of trees, but often achieves the highest test accuracy.

```python
for eta in [0.3, 0.1, 0.05, 0.01]:
    # Train and observe validation AUC curve...
```
*Result*: `eta = 0.1` provides stable, high validation AUC without requiring excessive boosting rounds.

#### Step 2: Tune `max_depth`
Unlike Random Forest (which uses deep trees of depth 10–20), gradient boosting performs best with shallow trees (depth 3–6):
- `max_depth = 3`: Simple rules, prevents complex feature co-adaptation, resistant to overfitting.
- `max_depth = 6`: Moderate complexity.
- `max_depth = 10`: Overfits rapidly; validation AUC drops after only a few dozen rounds.

*Result*: `max_depth = 3` produces the highest sustained validation AUC.

#### Step 3: Tune `min_child_weight`
`min_child_weight` is analogous to `min_samples_leaf` in Scikit-Learn: it specifies the minimum sum of instance weights required in a child node. Higher values enforce conservative splits:
- Candidate values: `[1, 10, 30]`.
- Higher `min_child_weight` smooths out noisy leaves in noisy datasets.

*Result*: `min_child_weight = 1` yields the best AUC ($\approx 0.836$).

#### Step 4: Determine Optimal `num_boost_round`
Plotting validation AUC against iteration number reveals the peak before overfitting begins:
- Around round $175$, validation AUC peaks at $\approx 0.836$.

---

### 9. Model Comparison & Final Evaluation

#### Comparing the Three Tuned Architectures on Validation Set:
| Model Family | Key Tuned Parameters | Validation AUC |
| :--- | :--- | :--- |
| **Decision Tree** | `max_depth=6`, `min_samples_leaf=15` | $\approx 0.785$ |
| **Random Forest** | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=3` | $\approx 0.825$ |
| **XGBoost** | `eta=0.1`, `max_depth=3`, `min_child_weight=1`, `rounds=175` | $\approx \mathbf{0.836}$ |

#### Engineering & Operational Trade-offs:
1. **Decision Tree**: Highly interpretable and fastest for inference, but underperforms ensembles significantly.
2. **Random Forest**: Strong accuracy, requires minimal hyperparameter tuning, virtually immune to overfitting by adding more trees, fully parallelizable during training.
3. **XGBoost**: Superior accuracy on tabular problems, but demands disciplined tuning, takes longer to configure, and can overfit if iteration count is uncontrolled.

#### Training the Final Selected Model
Following machine learning best practices:
1. Combine train and validation data into `df_full_train`.
2. Re-fit `DictVectorizer` on the full training partition.
3. Train the final XGBoost model using the best hyperparameters.
4. Evaluate once on the untouched test partition (`df_test`).

```python
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train.status == 'default').astype(int).values
del df_full_train['status']

dicts_full_train = df_full_train.fillna(0).to_dict(orient='records')
dicts_test = df_test.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=list(dv.get_feature_names_out()))
dtest = xgb.DMatrix(X_test, feature_names=list(dv.get_feature_names_out()))

final_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
}

final_model = xgb.train(final_params, dfulltrain, num_boost_round=175)

y_pred_test = final_model.predict(dtest)
test_auc = roc_auc_score(y_test, y_pred_test)
print(f"Final Test AUC: {test_auc:.4f}")
# Output: Final Test AUC: ~0.8323
```

> [!TIP]
> The test AUC ($0.8323$) is very close to the validation AUC ($0.8360$). The absence of a large discrepancy confirms that our tuning process did not overfit the validation set and that the model generalizes robustly.

---

### 10. Practical Considerations & Advanced Extensions

#### Feature Importance
Tree-based models provide built-in feature importance rankings:
- **Scikit-Learn (Random Forest)**: `rf.feature_importances_` measures Mean Decrease in Impurity (Gini importance).
- **XGBoost**: `model.get_score(importance_type='gain')` measures the average gain brought by a feature across all splits where it was used.
- Top predictive signals for credit default in this dataset include `records` (previous credit delinquency records), `job` (part-time vs fixed employment), and `seniority` (years with current employer).

#### Native Handling of Missing Values in XGBoost
Unlike Scikit-Learn's default trees, XGBoost natively handles missing values:
- During training, when a feature value is missing (`NaN`), XGBoost sends it to both the left and right child nodes and selects whichever direction yields the highest gain as the default branch.
- This eliminates the strict requirement for manual imputation (e.g., `fillna(0)`).

#### ExtraTrees (Extremely Randomized Trees)
`ExtraTreesClassifier` adds even more randomization than Random Forest:
- Instead of finding the optimal threshold $T^*$ among all possible thresholds for each candidate feature, it generates random candidate thresholds and selects the best among them.
- This further reduces variance, speeds up computation, and provides strong resistance against overfitting.

#### Tree Models for Regression
The same principles apply to regression tasks:
- **Decision Tree**: `DecisionTreeRegressor` (splits minimize MSE or MAE).
- **Random Forest**: `RandomForestRegressor`.
- **XGBoost**: Set `objective='reg:squarederror'` and `eval_metric='rmse'`.

---

## Summary (Module 6)
In Module 6, we mastered decision trees and ensemble learning architectures:
1. **Decision Trees**: Partition data using hierarchical if-then rules based on impurity reduction (Gini Impurity, Entropy). Without regularization (`max_depth`, `min_samples_leaf`), trees easily overfit by memorizing the training data.
2. **Random Forest**: An ensemble method that trains diverse decision trees independently in parallel using **Bagging** (bootstrap aggregation) combined with **random feature subsampling**. Random forests drastically reduce variance and do not overfit as more trees (`n_estimators`) are added.
3. **Gradient Boosting (XGBoost)**: An ensemble method that trains shallow trees sequentially, with each tree correcting the residual errors of the prior ensemble via gradient descent in function space.
4. **Hyperparameter Tuning**: Tuned `max_depth` and `min_samples_leaf` for Decision Trees and systematically optimized `eta`, `max_depth`, `min_child_weight`, and `num_boost_round` for XGBoost.
5. **Model Evaluation**: XGBoost achieved the highest validation performance ($AUC \approx 0.836$), which generalized consistently to unseen test data ($AUC \approx 0.832$), illustrating why gradient boosted trees are the dominant model family for structured tabular data.

---

## Glossary
- **Decision Tree**: A hierarchical non-parametric supervised learning model that splits data into partitions using if-then-else conditions.
- **Decision Stump**: A decision tree with a maximum depth of 1 (`max_depth=1`), performing only a single binary split.
- **Root Node**: The topmost node of a decision tree, representing the entire dataset prior to splitting.
- **Leaf (Terminal) Node**: An end node in a decision tree that outputs the final prediction and has no child nodes.
- **Gini Impurity**: A metric measuring the probability of incorrectly classifying a randomly chosen element if it were labeled according to class distribution: $Gini = 1 - \sum p_i^2$.
- **Entropy**: A measure of disorder or uncertainty in information theory: $H = -\sum p_i \log_2(p_i)$.
- **Information Gain**: The reduction in entropy or impurity achieved by partitioning data according to a specific feature split.
- **Overfitting**: A failure mode where a model learns training noise and specific sample peculiarities, degrading its performance on unseen data.
- **Pruning / Regularization**: Techniques (such as limiting `max_depth` or increasing `min_samples_leaf`) that prevent decision trees from growing excessively complex.
- **Ensemble Learning**: A meta-algorithm combining predictions from multiple individual models to improve accuracy and robustness.
- **Bagging (Bootstrap Aggregating)**: Ensembling technique where models are trained independently in parallel on bootstrap samples drawn with replacement.
- **Random Forest**: An ensemble method combining bagging of decision trees with random feature subsampling at each candidate split.
- **Out-Of-Bag (OOB) Samples**: The $\approx 36.8\%$ of training samples omitted from a given bootstrap sample, useful for internal validation.
- **Boosting**: An ensemble paradigm where models are trained sequentially, each attempting to correct the errors of previous models.
- **Gradient Boosting**: A boosting framework where each successive model is fitted to the negative gradients (pseudo-residuals) of the loss function.
- **Learning Rate ($\eta$ / `eta`)**: A shrinkage parameter scaling the contribution of each new tree in gradient boosting to control learning speed and prevent overfitting.
- **XGBoost (Extreme Gradient Boosting)**: An open-source, highly scalable, and optimized gradient boosting library.
- **DMatrix**: An internal, optimized binary data matrix structure used by XGBoost for memory efficiency and high-speed computation.
- **`min_child_weight`**: An XGBoost hyperparameter specifying the minimum sum of instance weights required in a child node, analogous to `min_samples_leaf`.
- **Feature Importance**: Quantitative score indicating how valuable each feature was in constructing the split nodes of an ensemble.
- **ExtraTrees**: Extremely Randomized Trees, an ensemble method that introduces additional randomness by picking random split thresholds rather than searching for optimal thresholds.
