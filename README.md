# Customer Churn Prediction with Logistic Regression from Scratch

This project presents a complete machine learning workflow for **customer churn prediction**, with a strong focus on the **mathematical foundations of optimization and logistic regression**.

The project is fully implemented in Python using NumPy, Pandas, and Matplotlib, with the classification model built **from scratch**.

It includes:

- mathematical study of gradient descent optimization;
- comparison of multiple learning rate schedules;
- exploratory data analysis (EDA);
- data preprocessing and feature engineering;
- logistic regression implementation from scratch;
- model evaluation and convergence analysis.



## Project Structure

```text
├── notebooks
│   ├── 01_gradient_descent_learning_rate_schedules.ipynb
│   ├── 02_telco_churn_eda.ipynb
│   └── 03_logistic_regression_from_scratch.ipynb
│
├── data
│   └── telco_churn_cleaned.csv
│
├── requirements.txt
└── README.md
```



## Notebook Overview

### 1. Optimization Foundations  
[`01_gradient_descent_learning_rate_schedules.ipynb`](./notebooks/01_gradient_descent_learning_rate_schedules.ipynb)


This notebook focuses on the mathematical and computational study of optimization algorithms.

It includes:

- implementation of gradient descent from scratch;
- comparison of multiple learning rate schedules;
- experiments on a differentiable convex quadratic function;
- convergence analysis;
- loss curve visualization;
- comparison of optimization speed and stability.

The test function is

$$
f(x)=x^T A x
$$

where $A$ is a diagonal positive definite matrix.

The learning rate schedules include:

- constant learning rate;
- step decay;
- inverse decay.

This notebook provides the theoretical foundation for the classification model used later in the project.



### 2. Exploratory Data Analysis  
[`02_telco_churn_eda.ipynb`](./notebooks/02_telco_churn_eda.ipynb)

This notebook performs exploratory data analysis on the customer churn dataset.

Main steps include:

- missing value analysis;
- target class distribution;
- categorical feature analysis;
- numerical feature distribution;
- churn behavior patterns;
- correlation analysis;
- dataset cleaning and preprocessing.

The output of this notebook is the cleaned dataset: [`telco_churn_cleaned.csv`](./data/telco_churn_cleaned.csv)



### 3. Logistic Regression from Scratch  
[`03_logistic_regression_from_scratch.ipynb`](./notebooks/03_logistic_regression_from_scratch.ipynb)

This notebook implements binary classification using logistic regression from scratch, including:

- sigmoid activation function;
- logistic loss (cross-entropy);
- gradient computation;
- gradient descent optimization;
- comparison of learning rate schedules;
- performance evaluation.



## Dataset

The project uses the **Telco Customer Churn dataset** from Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn



## Main Results

| Method | Final Loss | Accuracy | Recall |
|---|---:|---:|---:|
| Constant LR | 0.415969 | 0.7989 | 0.5668 |
| Step Decay | 0.468739 | 0.7775 | 0.6898 |
| Inverse Decay | 0.423077 | 0.7982 | 0.5882 |



## Key Insight

For customer churn prediction, **recall is especially important**, since failing to identify customers likely to leave may lead to direct business loss.

Therefore, **step decay** is selected as the preferred optimization strategy because it achieves the **highest recall**.

This highlights the importance of choosing optimization methods based not only on loss minimization, but also on task-specific business objectives.



## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook



## Mathematical Focus

This project emphasizes:

- optimization theory;
- gradient-based learning;
- logistic regression mathematics;
- practical ML implementation from scratch.

