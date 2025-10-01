# Bank Credit Scoring Model

![Credit Scoring](https://img.shields.io/badge/Machine%20Learning-Credit%20Scoring-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

This project builds a machine learning model to predict the probability of loan default by bank clients. Credit scoring models are crucial for financial institutions to assess credit risk and make informed lending decisions.

## Dataset

The dataset contains various financial and personal attributes of borrowers, including:

- Revolving credit utilization
- Age
- Payment history
- Debt ratio
- Monthly income
- Number of dependents
- And more...

The target variable indicates whether a borrower experienced a 90+ day delinquency.

## Features

- Comprehensive exploratory data analysis
- Data preprocessing and feature engineering
- Machine learning model development
- Model evaluation with multiple metrics
- Feature importance analysis

## Technologies Used

- **Python 3.8+**
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Analysis

Execute the Python script:

```bash
python credit_scoring.py
```

This will perform the complete analysis pipeline and generate visualization files in the current directory.

## Results

The model identifies key factors affecting loan default probability and provides a reliable credit scoring mechanism. Key insights include:

- Past payment delinquencies are strong predictors of future defaults
- Age and income level significantly impact default probability
- High debt ratios correlate with increased default risk

## Output Files

The script generates various visualization files including:
- Feature importance analysis
- ROC and precision-recall curves
- Confusion matrix
- Distribution plots of key variables

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset is derived from real financial records (anonymized)
- Special thanks to all contributors and reviewers

## Contact

For questions or feedback about this project, please open an issue on GitHub.
