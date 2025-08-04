# Credit Scoring Prediction for Home Credit

This project aims to build a predictive model that helps **Home Credit** evaluate the creditworthiness of loan applicants using a combination of financial, behavioral, and application data.

## Problem Statement

Home Credit needs a more accurate and automated way to assess credit risk. The goal is to predict whether a customer will default on a loan (`TARGET = 1`) or not.

## Dataset

Source: Home Credit Default Risk Competition (Kaggle)  
Datasets used:
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`

All datasets were joined and preprocessed into a single dataset: `df_final_merged_all.csv`.

## Exploratory Data Analysis

- Distribution of key features like income, loan amount, and late payment history.
- Analysis of default rate by occupation, education, and payment behavior.
- Detected class imbalance in the target variable.

## Feature Engineering

- One-hot encoding for categorical variables.
- Creation of new ratio-based features (e.g., payment vs installment).
- Feature selection using `SelectKBest` and model-based importance.

## Modeling

Models used:
- **Logistic Regression** (baseline)
- **XGBoost** (default, with class_weight, and with scale_pos_weight)

Balancing Techniques:
- Class weights
- SMOTE (not final model)

### Evaluation Metrics

- **Accuracy**
- **Recall (class 1)**
- **F1-Score**
- **AUC-ROC**
- Tested various thresholds (0.5, 0.4, 0.3)

## Results Summary

| Model                     | Accuracy | Recall (1) | F1-Score (1) | AUC  |
|--------------------------|----------|------------|--------------|------|
| Logistic Regression      | 0.92     | 0.012      | 0.023        | 0.57 |
| XGBoost (default)        | 0.71     | 0.66       | 0.27         | 0.75 |
| XGBoost (class_weight)   | 0.58     | 0.80       | 0.23         | 0.74 |

## Business Insights

- Government employees (PNS) have higher approval rates but are underrepresented.
- Payment discipline features are strong predictors of loan default.
- XGBoost with class weighting provides better recall for detecting defaults.

## Tech Stack

- Python (Pandas, Scikit-Learn, XGBoost)
- Jupyter Notebook
- Matplotlib & Seaborn
- Git & GitHub



