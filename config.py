"""
Configuration file for Credit Scoring Model
"""

# Data generation settings
DATA_CONFIG = {
    'n_samples': 10000,
    'random_state': 42,
    'test_size': 0.3
}

# Model parameters
MODEL_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42
    },
    'decision_tree': {
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42
    },
    'gradient_boosting': {
        'random_state': 42
    }
}

# Hyperparameter tuning
HYPERPARAM_CONFIG = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Business parameters
BUSINESS_CONFIG = {
    'approval_rate': 0.7,
    'avg_loan_amount': 10000,
    'loss_given_default': 0.5,
    'profit_margin': 0.1
}