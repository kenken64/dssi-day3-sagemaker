# Employee Salary Prediction with AWS SageMaker

A machine learning project that predicts employee salaries based on years of experience, demonstrating both local scikit-learn training and AWS SageMaker managed training and deployment.

---

## Overview

This notebook walks through a complete ML pipeline:
- Exploratory data analysis on a salary dataset
- Local model training with scikit-learn Linear Regression
- Cloud-based training with the AWS SageMaker Linear Learner algorithm
- Real-time inference via a deployed SageMaker endpoint

---

## Dataset

**File**: `salary.csv`

| Column | Type | Description |
|---|---|---|
| `YearsExperience` | float | Independent variable (X) — 1.1 to 13.5 years |
| `Salary` | int | Dependent variable (Y) — $37,731 to $139,465 |

- **35 samples** total
- Mean experience: **6.31 years** | Mean salary: **$83,945.60**
- No null values

---

## Setup: AWS SageMaker

1. Sign in to your AWS account and navigate to the **SageMaker** service page.
2. Create a **Notebook Instance**.
3. Create an **Amazon SageMaker IAM role** with S3 bucket access (`AmazonSageMaker-ExecutionRole-*`).
4. Wait (~2 minutes) for the instance to reach **InService** status.
5. Open **Jupyter Notebook** from the instance.

---

## Dependencies

```python
# Install if not already available
# !pip install seaborn tensorflow

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sagemaker
```

---

## Notebook Walkthrough

### 1. Data Exploration
- Load `salary.csv` into a Pandas DataFrame
- Inspect with `.head()`, `.tail()`, `.info()`, `.describe()`
- Visualise data using scatter plots (matplotlib / seaborn)
- Confirm no null values via heatmap

### 2. Train/Test Split
- 75% training (~26 samples), 25% test (~9 samples)
- Using `sklearn.model_selection.train_test_split`

### 3. Local Model — scikit-learn Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
```

**Results:**
- R² Score: **0.9815** (98.15%)
- Slope (m): **8,714.14**
- Intercept (b): **29,703.00**

### 4. AWS SageMaker — Linear Learner

Training data is serialised to RecordIO format and uploaded to S3, then the SageMaker built-in Linear Learner algorithm is used.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `feature_dim` | 1 |
| `predictor_type` | `regressor` |
| `mini_batch_size` | 5 |
| `epochs` | 5 |
| `num_models` | 32 |
| `loss` | `absolute_loss` |

**Training config:**
- Instance type: `ml.c4.xlarge`
- S3 bucket: `sagemaker-practical-kpty`
- Training time: ~152 seconds

**Evaluation metrics on test set:**

| Metric | Value | What it means |
|---|---|---|
| MAE (Mean Absolute Error) | $5,372.01 | On average, predictions are off by $5,372 from the actual salary. Every error counts equally. |
| RMSE (Root Mean Squared Error) | $6,451.44 | Similar to MAE but penalises large errors more heavily. Higher than MAE, indicating a few predictions were significantly off. |
| MSE (Mean Squared Error) | 41,621,111.69 | Average of squared errors (units are dollars²). √MSE = RMSE. Used as the loss function during training, not for direct interpretation. |

> The salary range in the dataset is $37,731–$139,465 (spread of ~$101,734). An MAE of $5,372 is within ~5% of that range — reasonably accurate for a single-feature linear model.

### 5. Deployment & Inference

```python
linear_regressor = linear.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

# Configure serializer/deserializer
linear_regressor.serializer = CSVSerializer()
linear_regressor.deserializer = JSONDeserializer()

# Predict
predictions = linear_regressor.predict(X_test)
# Returns: {'predictions': [{'score': 113986.77}, ...]}
```

Endpoint name example: `linear-learner-2024-03-15-18-54-36-038`

---

## Project Structure

```
dssi-day3-sagemaker/
├── employee_salary_prediction_notebook.ipynb   # Main notebook
├── employee_salary_prediction_notebook _w.ipynb # Working copy
├── salary.csv                                   # Dataset
└── README.md
```

---

## Key Takeaways

- The scikit-learn model achieves **98.15% R²**, showing a near-perfect linear relationship between years of experience and salary.
- The SageMaker Linear Learner trains 32 candidate models in parallel and selects the best, achieving an RMSE of ~$6,451.
- The notebook demonstrates end-to-end MLOps: local experimentation → cloud training → live endpoint deployment.
