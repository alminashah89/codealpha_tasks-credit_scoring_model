import pandas as pd
import numpy as np
import json
from datetime import datetime

def save_model_artifacts(model, preprocessor, feature_names, filename_prefix='credit_model'):
    """Save model artifacts for later use"""
    import joblib
    
    artifacts = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(artifacts, f'{filename_prefix}_artifacts.pkl')
    print(f"Model artifacts saved to {filename_prefix}_artifacts.pkl")

def load_model_artifacts(filename='credit_model_artifacts.pkl'):
    """Load saved model artifacts"""
    import joblib
    return joblib.load(filename)

def create_sample_applicant():
    """Create a sample applicant for testing"""
    return {
        'age': 35,
        'income': 75000,
        'debt_to_income_ratio': 25.5,
        'credit_utilization': 30.2,
        'payment_delinquencies': 0,
        'late_payments_30d': 1,
        'late_payments_90d': 0,
        'credit_history_length': 12,
        'number_of_credit_cards': 3,
        'total_debt': 25000,
        'employment_length': 8,
        'income_to_debt_ratio': 3.0,
        'credit_age_ratio': 0.34,
        'total_late_payments': 1,
        'payment_risk_score': 1,
        'debt_risk_score': 55.7,
        'income_category': 'High',
        'dti_category': 'Excellent'
    }

def validate_applicant_data(applicant_data, required_features):
    """Validate that applicant data has all required features"""
    missing_features = [feature for feature in required_features if feature not in applicant_data]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    return True