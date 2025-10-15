import pandas as pd
import numpy as np
from config import DATA_CONFIG

class CreditDataGenerator:
    def __init__(self, n_samples=DATA_CONFIG['n_samples'], random_state=DATA_CONFIG['random_state']):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_synthetic_data(self):
        """Generate synthetic credit data for demonstration"""
        
        # Basic demographic and financial features
        age = np.random.normal(45, 15, self.n_samples).astype(int)
        age = np.clip(age, 18, 80)
        
        income = np.random.lognormal(10.5, 0.8, self.n_samples)
        income = np.clip(income, 15000, 200000)
        
        debt_to_income = np.random.beta(2, 5, self.n_samples) * 100
        credit_utilization = np.random.beta(3, 3, self.n_samples) * 100
        
        # Payment history features
        payment_delinquencies = np.random.poisson(0.3, self.n_samples)
        late_payments_30d = np.random.poisson(0.5, self.n_samples)
        late_payments_90d = np.random.poisson(0.1, self.n_samples)
        
        # Credit history features
        credit_history_length = np.random.normal(15, 8, self.n_samples)
        credit_history_length = np.clip(credit_history_length, 0, 50)
        
        number_of_credit_cards = np.random.poisson(3, self.n_samples)
        total_debt = income * (debt_to_income / 100) * np.random.uniform(0.8, 1.2, self.n_samples)
        
        # Employment features
        employment_length = np.random.normal(8, 6, self.n_samples)
        employment_length = np.clip(employment_length, 0, 40)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'income': income,
            'debt_to_income_ratio': debt_to_income,
            'credit_utilization': credit_utilization,
            'payment_delinquencies': payment_delinquencies,
            'late_payments_30d': late_payments_30d,
            'late_payments_90d': late_payments_90d,
            'credit_history_length': credit_history_length,
            'number_of_credit_cards': number_of_credit_cards,
            'total_debt': total_debt,
            'employment_length': employment_length
        })
        
        # Generate target variable (creditworthiness)
        risk_score = (
            data['debt_to_income_ratio'] * 0.3 +
            data['payment_delinquencies'] * 0.25 +
            data['late_payments_90d'] * 0.2 +
            data['credit_utilization'] * 0.15 -
            data['income'] * 0.00001 +
            np.random.normal(0, 1, self.n_samples)
        )
        
        # Convert to binary classification (0: Good, 1: Bad)
        data['credit_risk'] = (risk_score > risk_score.mean()).astype(int)
        
        return data

    def save_data(self, data, filename='credit_data.csv'):
        """Save generated data to CSV"""
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def load_data(self, filename='credit_data.csv'):
        """Load data from CSV"""
        return pd.read_csv(filename)