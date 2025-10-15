from scoring_system import CreditScoringSystem
from utils import load_model_artifacts

# Load the trained model
artifacts = load_model_artifacts()
scoring_system = CreditScoringSystem(
    artifacts['model'], 
    artifacts['preprocessor'], 
    artifacts['feature_names']
)

# Test applicant with POOR credit
poor_applicant = {
    'age': 45, 
    'income': 50000,
    'debt_to_income_ratio': 60.0,  # Very high debt
    'credit_utilization': 80.0,    # High credit usage
    'payment_delinquencies': 3,    # Multiple delinquencies
    'late_payments_30d': 5,        # Many late payments
    'late_payments_90d': 2,        # Serious late payments
    'credit_history_length': 8,
    'number_of_credit_cards': 6,
    'total_debt': 35000,
    'employment_length': 3,
    'income_to_debt_ratio': 1.43,
    'credit_age_ratio': 0.18,
    'total_late_payments': 9,
    'payment_risk_score': 12,
    'debt_risk_score': 140.0,
    'income_category': 'Medium',
    'dti_category': 'Poor'
}

print("=== TESTING POOR CREDIT APPLICANT ===")
result1 = scoring_system.generate_credit_report(poor_applicant)

print("\n" + "="*50 + "\n")

# Test applicant with EXCELLENT credit
excellent_applicant = {
    'age': 35,
    'income': 85000,
    'debt_to_income_ratio': 15.0,  # Low debt
    'credit_utilization': 20.0,    # Low credit usage  
    'payment_delinquencies': 0,    # No delinquencies
    'late_payments_30d': 0,        # No late payments
    'late_payments_90d': 0,
    'credit_history_length': 12,
    'number_of_credit_cards': 3,
    'total_debt': 15000,
    'employment_length': 8,
    'income_to_debt_ratio': 5.67,
    'credit_age_ratio': 0.34,
    'total_late_payments': 0,
    'payment_risk_score': 0,
    'debt_risk_score': 35.0,
    'income_category': 'High', 
    'dti_category': 'Excellent'
}

print("=== TESTING EXCELLENT CREDIT APPLICANT ===")
result2 = scoring_system.generate_credit_report(excellent_applicant)