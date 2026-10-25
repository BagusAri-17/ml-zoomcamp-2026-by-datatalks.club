# Homework 5: Deploying Machine Learning Models

📓 Full solution notebook: [homework.ipynb](./homework.ipynb)

---

## Questions & Answers

### Q1. Environment Check

Run `uv --version` and record the version in your notes.

```bash
uv --version
```

**Answer: `uv 0.13.5`**

---

### Q2. Locked Dependency

Open `pyproject.toml` and `uv.lock`. Which Scikit-Learn version is part of the 2026 reference environment?

- `1.6.1`
- **`1.7.2`** ✅
- `1.8.0`
- `2.0.0`

**Answer: `1.7.2`**

---

### Q3. Load the Model

Write a Python script that loads `pipeline.bin` with `pickle` and calls `predict_proba` for this lead:

```json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```

What is the probability of conversion (rounded to three decimal places)?

**Answer: `0.533` (exact: `0.532994`)**

---

### Q4. Serve the Model

Start the provided API with `uv run uvicorn predict:app --host 0.0.0.0 --port 9696`. Send this lead to the `/predict` endpoint:

```json
{
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}
```

What is the `conversion_probability` returned by the API (rounded to three decimal places)?

**Answer: `0.770` (exact: `0.769799`)**

---

### Q5. Container Configuration

Which Python base-image tag is declared in the `Dockerfile`?

- `python:3.9-slim-bullseye`
- **`python:3.11.15-slim-bookworm`** ✅
- `python:3.12-slim-bookworm`
- `python:3.13.10-slim-bookworm`

**Answer: `python:3.11.15-slim-bookworm`**

---

### Q6. Run the Container

Run the same request from Question 4 against the Docker container (`zoomcamp-model:2026-hw5`). What is the `conversion_probability` rounded to three decimal places?

**Answer: `0.770` (exact: `0.769799`)**
