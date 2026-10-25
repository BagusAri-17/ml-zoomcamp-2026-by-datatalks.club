# Homework 6: Decision Trees and Ensemble Learning

📓 Full solution notebook: [homework.ipynb](./homework.ipynb)

---

## Questions & Answers

### Q1. Original Splitting Feature

Let's train a decision tree regressor to predict the `fuel_efficiency_mpg` variable. Train a model with `max_depth=1`.

Which original feature is used for splitting the data? If you use `DictVectorizer`, map a one-hot feature such as `origin=USA` back to its original column name.

- `'vehicle_weight'`
- **`'model_year'`** ✅
- `'origin'`
- `'fuel_type'`

**Answer: `'model_year'`**

> **Explanation**:
> Using `DecisionTreeRegressor(max_depth=1)` and inspecting the split via `export_text`:
> ```text
> |--- model_year <= 1997.50
> |   |--- value: [28.61]
> |--- model_year >  1997.50
> |   |--- value: [31.17]
> ```
> The root node splits on the feature `model_year` at threshold `1997.50`.

---

### Q2. Training the Random Forest Regressor

Train a random forest regressor with these parameters:
- `n_estimators=10`
- `random_state=1`
- `n_jobs=-1` (optional - to make training faster)

What's the RMSE of this model on the validation data?

- 0.1837
- **1.837** ✅
- 18.37
- 183.7

**Answer: `1.837` (Validation RMSE is 1.8394)**

> **Explanation**:
> Computing the Root Mean Squared Error on the validation set:
> ```python
> rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
> rf.fit(X_train, y_train)
> y_pred = rf.predict(X_val)
> root_mean_squared_error(y_val, y_pred)
> # Output: 1.83938... ≈ 1.837
> ```

---

### Q3. Experimenting with `n_estimators`

Now let's experiment with the `n_estimators` parameter:
- Try these values: `[10, 50, 100, 150]`
- Set `random_state=1`
- Evaluate the model on the validation dataset

Which value gives the lowest validation RMSE? Consider 3 decimal places for calculating the answer.

- 10
- 50
- **100** ✅
- 150

**Answer: `100` (Validation RMSE is 1.768, lowest among the tested values)**

> **Explanation**:
> Evaluating validation RMSE for each value of `n_estimators`:
> - `n_estimators = 10`: RMSE = `1.839` (1.839383)
> - `n_estimators = 50`: RMSE = `1.773` (1.772863)
> - `n_estimators = 100`: RMSE = `1.768` (1.767643) 👈 *Lowest RMSE*
> - `n_estimators = 150`: RMSE = `1.768` (1.768109)
> 
> Both `100` and `150` round to `1.768` at 3 decimal places, with `100` achieving the absolute lowest RMSE at `1.7676`.

---

### Q4. Selecting the Best `max_depth`

Let's select the best `max_depth`:
- Try different values of `max_depth`: `[10, 15, 20, 25]`
- For each of these values,
  - try different values of `n_estimators` from `[10, 50, 100, 150]`
  - calculate the mean RMSE
- Fix the random seed: `random_state=1`

What's the best `max_depth`, using the mean RMSE?

- **10** ✅
- 15
- 20
- 25

**Answer: `10` (Mean validation RMSE across n_estimators is 1.7541)**

> **Explanation**:
> Calculating the mean RMSE across `n_estimators` for each depth:
> - `max_depth = 10`: Mean RMSE = **`1.7541`** 👈 *Best / Lowest Mean RMSE*
> - `max_depth = 15`: Mean RMSE = `1.7837`
> - `max_depth = 20`: Mean RMSE = `1.7878`
> - `max_depth = 25`: Mean RMSE = `1.7867`
> 
> Restricting `max_depth` to 10 regularizes the individual trees, preventing them from overfitting the training data and achieving the best validation performance.

---

### Q5. Feature Importance

We can extract feature importance information from tree-based models via `feature_importances_`.
- Train the model with these parameters:
  - `n_estimators=10`
  - `max_depth=20`
  - `random_state=1`
  - `n_jobs=-1`
- Get the feature importance information from this model.

What's the most important feature (among these 4)?

- **`vehicle_weight`** ✅
- `horsepower`
- `acceleration`
- `engine_displacement`

**Answer: `vehicle_weight` (Feature importance is 0.2031)**

> **Explanation**:
> Inspecting the top features from `rf.feature_importances_`:
> 1. `model_year`: `0.2959`
> 2. **`vehicle_weight`**: **`0.2031`** 👈 *Highest among the 4 options*
> 3. `horsepower`: `0.0785`
> 4. `engine_displacement`: `0.0612`
> 5. `acceleration`: `0.0553`
> 
> Among the four candidate choices provided in the question, `vehicle_weight` has the highest importance.

---

### Q6. Tuning XGBoost `eta`

Train an XGBoost model with these parameters for 100 rounds:
```python
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
```
Now change `eta` from `0.3` to `0.1`.

Which eta leads to the best RMSE score on the validation dataset?

- 0.3
- **0.1** ✅
- Both give equal value

**Answer: `0.1` (Validation RMSE is 1.7248 for eta=0.1 vs 1.8265 for eta=0.3)**

> **Explanation**:
> Comparing the two learning rates across 100 boosting iterations:
> - `eta = 0.3`: Validation RMSE at round 100 is **`1.8265`** (min validation RMSE is `1.7320` at earlier rounds before overfitting).
> - `eta = 0.1`: Validation RMSE at round 100 is **`1.7248`** (min validation RMSE is `1.7122`).
> 
> A lower learning rate (`eta = 0.1`) shrinks tree contributions more effectively, preventing premature overfitting on the training data and achieving lower validation error.
