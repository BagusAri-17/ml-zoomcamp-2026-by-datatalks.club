# Homework 2: Machine Learning for Regression

📓 Full solution notebook: [homework.ipynb](./homework.ipynb)

---

## Questions & Answers

### Q1. Column with missing values

There's one column with missing values. What is it?

- `'engine_displacement'`
- **`'horsepower'`** ✅
- `'vehicle_weight'`
- `'model_year'`

**Answer: `'horsepower'`**

---

### Q2. Median for variable 'horsepower'

What's the median (50% percentile) for variable `'horsepower'`?

- 204
- **254** ✅
- 304
- 354

**Answer: `254`**

---

### Q3. Best option for filling NAs

Which option gives better RMSE when filling missing values (evaluating on the validation dataset without regularization)?

- With 0
- **With mean** ✅
- Both are equally good

**Answer: `With mean` (RMSE with mean is 2.202, RMSE with 0 is 2.205)**

---

### Q4. Best regularization parameter r

Which `r` gives the best RMSE on the validation dataset? 

- **0** ✅
- 0.01
- 0.1
- 1
- 5
- 10
- 100

**Answer: `0` (RMSE is 2.2053)**

---

### Q5. STD of RMSE scores for different seeds

What's the standard deviation of all the scores using seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`?

- 0.006
- 0.016
- **0.029** ✅
- 0.036

**Answer: `0.029`**

---

### Q6. RMSE on test dataset

What's the RMSE on the test dataset using seed `9`, combined train and validation sets, and `r=0.001`?

- 0.236
- **2.236** ✅
- 22.10
- 221.0

**Answer: `2.236`**
