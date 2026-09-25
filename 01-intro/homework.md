# Homework 1: Introduction to Machine Learning

📓 Full solution notebook: [homework.ipynb](./homework.ipynb)

---

## Questions & Answers

### Q1. Pandas Version

What version of Pandas did you install?

```python
pd.__version__
```

**Answer: `3.0.6`**

---

### Q2. Records Count

How many records are in the dataset?

- 5000
- 9000
- **10000** ✅
- 15000

**Answer: `10000`**

---

### Q3. Fuel Types

How many fuel types are presented in the dataset?

- 1
- 2
- **3** ✅
- 4

**Answer: `3` — `['Gasoline', 'Diesel', 'Hybrid']`**

---

### Q4. Missing Values

How many columns in the dataset have missing values?

- 0
- 1
- **2** ✅
- 3
- 4

**Answer: `2`**

---

### Q5. Max Fuel Efficiency

What's the maximum fuel efficiency of cars from Asia?

- 21.2
- 31.2
- **41.2** ✅
- 51.2

**Answer: `41.2`**

---

### Q6. Median Value of Horsepower

1. Find the median value of the `horsepower` column.
2. Calculate the most frequent value of `horsepower`.
3. Fill missing values using `fillna` with the most frequent value.
4. Recalculate the median.

Has it changed?

- Yes, it increased
- Yes, it decreased
- **No** ✅

**Answer: `No`** — Median before: `254.0` | Mode used to fill: `252.0` | Median after: `254.0`

---

### Q7. Sum of Weights

1. Select all cars from Asia
2. Select columns `vehicle_weight` and `model_year`
3. Take the first 7 rows → matrix `X`
4. Compute `XTX = X.T @ X`
5. Invert `XTX`
6. Create `y = [1100, 1300, 800, 900, 1000, 1100, 1200]`
7. Compute `w = XTX⁻¹ @ X.T @ y`
8. Sum all elements of `w`

- 0.0369
- **0.369** ✅
- 3.69
- 36.9

**Answer: `0.36919696904925203 or 0.369`**
