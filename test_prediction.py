from scoring_system import CreditScoringSystem
from utils import load_model_artifacts

# Load saved model
artifacts = load_model_artifacts('credit_model_artifacts.pkl')
scoring_system = CreditScoringSystem(
    artifacts['model'],
    artifacts['preprocessor'],
    artifacts['feature_names']
)

# Test applicant
applicant_data = {
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

result = scoring_system.predict_credit_risk(applicant_data)
print("=== CREDIT REPORT ===")
for key, value in result.items():
    print(f"{key}: {value}")