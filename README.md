# Bank Credit Scoring Model

![Credit Scoring](https://img.shields.io/badge/Machine%20Learning-Credit%20Scoring-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
A machine learning pipeline to predict loan default probability, with:
- A CLI script for full analysis and artifact generation
- A Streamlit dashboard for interactive EDA, model training, metrics, and live predictions

## Project Structure
- credit_scoring.py — CLI pipeline: EDA, preprocessing, training, metrics, plots
- credit_scoring_app.py — Streamlit dashboard (upload CSV → explore → train → evaluate → predict)
- CreditScoring.ipynb — Original notebook (will be removed later)

## Dataset
Typical columns:
- SeriousDlqin2yrs (target: 0/1)
- age, DebtRatio, MonthlyIncome, NumberOfDependents
- Payment delinquency counts, utilization metrics, etc.

## Features
- EDA: distributions, correlation matrix
- Preprocessing: missing values, basic encoding for categoricals
- Modeling: Logistic Regression baseline
- Evaluation: confusion matrix, ROC AUC, PR curve, classification report
- Feature importance (coefficients)
- Streamlit app: upload data, interactively analyze and predict

## Requirements
```bash
pip install -r requirements.txt
```
Main libraries: numpy, pandas, seaborn, matplotlib, scikit-learn, streamlit.

## Dependencies and requirements.txt
Keep and commit requirements.txt for reproducible installs (local, CI, deployment).

- Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

- Install dependencies from the repo file:
```bash
pip install -r requirements.txt
```

- Update requirements.txt after changing packages:
```bash
pip freeze > requirements.txt
```

## Quick Start

CLI (generates plots and metrics to files):
```bash
python credit_scoring.py
```

Streamlit app:
```bash
streamlit run credit_scoring_app.py
```

## Using the Streamlit App
1. Upload a CSV containing SeriousDlqin2yrs and features.
2. Explore: target distribution, numeric distributions, correlation matrix.
3. Train: one-click Logistic Regression with stratified split.
4. Evaluate: confusion matrix, ROC AUC, classification report, top features.
5. Predict: input feature values and get default probability with a simple recommendation.

## Results (CLI)
The script saves:
- confusion_matrix.png
- roc_curve.png
- precision_recall_curve.png
- feature_importance.png
- age_distribution.png, debt_ratio_distribution.png, numerical_distributions.png, correlation_matrix.png

## License
MIT — see LICENSE.

## Contact
Open an issue on GitHub for questions or suggestions.
