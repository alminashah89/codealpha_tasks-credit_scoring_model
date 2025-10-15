import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CreditDataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.features = data.drop('credit_risk', axis=1)
        self.target = data['credit_risk']
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        
        print("=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print("\n1. Basic Statistics:")
        print(self.data.describe())
        
        # Check for missing values
        print("\n2. Missing Values:")
        print(self.data.isnull().sum())
        
        # Correlation analysis
        print("\n3. Correlation with Target:")
        correlations = self.data.corr()['credit_risk'].sort_values(ascending=False)
        print(correlations)
        
        # Visualize correlations
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def create_new_features(self):
        """Create new engineered features"""
        
        print("\n=== Feature Engineering ===")
        
        # Create interaction features
        self.data['income_to_debt_ratio'] = self.data['income'] / (self.data['total_debt'] + 1)
        self.data['credit_age_ratio'] = self.data['credit_history_length'] / (self.data['age'] + 1)
        self.data['total_late_payments'] = (self.data['late_payments_30d'] + 
                                          self.data['late_payments_90d'] * 2)
        
        # Create risk score features
        self.data['payment_risk_score'] = (self.data['payment_delinquencies'] + 
                                         self.data['total_late_payments'])
        
        self.data['debt_risk_score'] = (self.data['debt_to_income_ratio'] + 
                                      self.data['credit_utilization'])
        
        # Create categorical features from continuous variables
        self.data['income_category'] = pd.cut(self.data['income'], 
                                            bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        self.data['dti_category'] = pd.cut(self.data['debt_to_income_ratio'],
                                         bins=[0, 20, 40, 60, 100],
                                         labels=['Excellent', 'Good', 'Fair', 'Poor'])
        
        print("New features created:")
        new_features = [col for col in self.data.columns if col not in ['credit_risk'] and col not in self.features.columns]
        print(new_features)
        
        return self.data

    def plot_target_distribution(self):
        """Plot distribution of target variable"""
        plt.figure(figsize=(8, 6))
        self.target.value_counts().plot(kind='bar')
        plt.title('Distribution of Credit Risk')
        plt.xlabel('Credit Risk (0: Good, 1: Bad)')
        plt.ylabel('Count')
        plt.show()