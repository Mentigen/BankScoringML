"""
Bank Credit Scoring: Predicting Loan Default

This script builds a machine learning model to predict the probability of loan default by bank clients.
The goal is to help financial institutions assess credit risk and make informed lending decisions.
"""

# --- Import necessary libraries ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# Set plot styling for better visualization
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Constants
RANDOM_STATE = 42
DATASET_PATH = "~/development/ML/datasets/credit_scoring.csv"

def create_plot(data=None, plot_type='hist', x=None, y=None, hue=None, title=None,
                xlabel=None, ylabel=None, color='steelblue', filename=None):
    """Helper function to create and save plots with consistent styling"""
    plt.figure(figsize=(12, 6))

    if plot_type == 'hist':
        sns.histplot(data=data, x=x, kde=True, color=color)
    elif plot_type == 'bar':
        if hue is not None:
            sns.barplot(data=data, x=x, y=y, hue=hue, palette=['skyblue', 'salmon'])
        else:
            sns.barplot(data=data, x=x, y=y, color='skyblue')
    elif plot_type == 'count':
        if hue is not None:
            ax = sns.countplot(data=data, x=x, hue=hue, palette=['skyblue', 'salmon'])
        else:
            ax = sns.countplot(data=data, x=x, hue=x, palette=['skyblue', 'salmon'], legend=False)
        if hue is None:
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / len(data):.1f}%'
                ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=12)
    elif plot_type == 'heat':
        sns.heatmap(data, annot=True, fmt="d", cmap="Blues", cbar=False)

    if title:
        plt.title(title, fontsize=16)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def load_and_explore_data():
    """Load the dataset and perform initial exploration"""
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Display basic dataset information
    print("Dataset Information:")
    print(df.info())
    print("\nSample data:")
    print(df.sample(3))
    print("\nNumerical features summary:")
    print(df.describe())
    
    if any(df.dtypes == 'object'):
        print("\nCategorical features summary:")
        print(df.describe(include='object'))

    return df

def check_data_quality(df):
    """Check for missing values and duplicates"""
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) > 0:
        print("Missing values by column:")
        print(missing_values)
    else:
        print("No missing values found.")

    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    return missing_values

def exploratory_data_analysis(df):
    """Perform exploratory data analysis with visualizations"""
    # Check data quality
    missing_values = check_data_quality(df)

    # Target variable distribution
    print("Target Variable Distribution:")
    print(df['SeriousDlqin2yrs'].value_counts())
    print("\nTarget Variable Distribution (%):")
    print(df['SeriousDlqin2yrs'].value_counts(normalize=True).round(4) * 100)
    
    # Plot distributions of key variables
    create_plot(data=df, x='age', color='steelblue',
                title='Distribution of Customer Age', xlabel='Age (years)',
                ylabel='Frequency', filename='age_distribution.png')

    create_plot(data=df, x='DebtRatio', color='darkred',
                title='Distribution of Debt Ratio', xlabel='Debt Ratio',
                ylabel='Frequency', filename='debt_ratio_distribution.png')

    create_plot(data=df, x='SeriousDlqin2yrs', plot_type='count',
                title='Distribution of Target Variable (Default)',
                xlabel='Default Status (1 = Default, 0 = No Default)',
                ylabel='Count', filename='target_distribution.png')

    # Visualize numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    # Create subplots for numerical features
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(16, n_rows * 4))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[feature], kde=True, color='darkblue', alpha=0.7)
        plt.title(f'Distribution of {feature}', fontsize=12)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    plt.close()
    
    # Create correlation matrix
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(14, 12))
    correlation_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Analyze categorical features if any
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        create_plot(data=df, x=feature, hue='SeriousDlqin2yrs', plot_type='count',
                    title=f'{feature} vs Default Status', xlabel=feature,
                    ylabel='Count', filename=f'{feature}_vs_default.png')

def preprocess_data(df, missing_values=None):
    """Handle missing values and encode categorical variables"""
    df_processed = df.copy()

    # Handle missing values
    if missing_values is not None and len(missing_values) > 0:
        for col in missing_values.index:
            if df[col].dtype in ['int64', 'float64']:
                fill_value = df[col].mean()
                # Avoid chained assignment warning
                df_processed[col] = df_processed[col].fillna(fill_value)
                print(f"Filled missing values in {col} with mean: {fill_value:.2f}")
            else:
                fill_value = df[col].mode()[0]
                df_processed[col] = df_processed[col].fillna(fill_value)
                print(f"Filled missing values in {col} with mode: {fill_value}")

    # Feature analysis with target
    create_plot(data=df_processed, x='SeriousDlqin2yrs', y='age', plot_type='bar',
                title='Average Age by Default Status',
                xlabel='Default Status (1 = Default, 0 = No Default)',
                ylabel='Average Age', filename='age_vs_default.png')

    # Encode categorical variables
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col == 'RealEstateLoansOrLines':
            # Create a mapping from letters to numbers (A=1, B=2, etc.)
            mapping = {chr(65+i): i+1 for i in range(26)}
            df_processed[f'{col}_numeric'] = df_processed[col].map(mapping)
            print(f"Encoded {col}")
        else:
            # Generic encoding for other categorical variables
            unique_values = df_processed[col].unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
            df_processed[f'{col}_numeric'] = df_processed[col].map(mapping)
            print(f"Encoded {col}")
            print(f"Mapping: {mapping}")

    return df_processed

def prepare_features(df):
    """Prepare features and target variable for modeling"""
    # Identify original categorical columns that have been encoded
    original_categorical = [col for col in df.columns if col + '_numeric' in df.columns]

    # Prepare features (X) and target variable (y)
    X = df.drop(columns=['SeriousDlqin2yrs'] + original_categorical)
    y = df['SeriousDlqin2yrs']

    return X, y

def train_and_evaluate_model(X, y):
    """Train model and evaluate performance"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Print sample predictions
    print("First 10 predictions (0 = No Default, 1 = Default):")
    print(y_pred[:10])
    
    # Evaluate model performance
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Create a heatmap for the confusion matrix
    create_plot(data=cm, plot_type='heat',
                title='Confusion Matrix', xlabel='Predicted Label',
                ylabel='True Label', filename='confusion_matrix.png')

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # Feature importance analysis
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })

    coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

    plt.figure(figsize=(12, 8))
    # Remove palette if no hue is set to avoid FutureWarning
    sns.barplot(x='Coefficient', y='Feature', data=coefficients, color='mediumseagreen')
    plt.title('Feature Importance (Logistic Regression Coefficients)', fontsize=16)
    plt.xlabel('Coefficient Value', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('feature_importance.png')
    plt.close()

    return {
        'model': model,
        'coefficients': coefficients,
        'performance': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
        }
    }

def main():
    """Main function to execute the analysis pipeline"""
    print("Starting Credit Scoring Analysis...")
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Perform exploratory data analysis
    print("\nPerforming exploratory data analysis...")
    exploratory_data_analysis(df)
    
    # Check missing values before preprocessing
    missing_values = check_data_quality(df)

    # Preprocess data
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df, missing_values)

    # Prepare features
    print("\nPreparing features for modeling...")
    X, y = prepare_features(df_processed)
    print("Selected features for modeling:")
    print(X.columns.tolist())

    # Train and evaluate model
    print("\nTraining and evaluating model...")
    results = train_and_evaluate_model(X, y)

    # Display top features
    print("\nTop 5 most important features:")
    print(results['coefficients'][['Feature', 'Coefficient']].head(5))

    print("\nAnalysis completed successfully!")
    print("Results and visualizations saved to the current directory.")

if __name__ == "__main__":
    main()
