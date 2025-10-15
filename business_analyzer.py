import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import BUSINESS_CONFIG

class BusinessImpactAnalyzer:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def calculate_business_metrics(self, approval_rate=0.7, avg_loan_amount=10000, 
                                 loss_given_default=0.5, profit_margin=0.1):
        """Calculate business impact metrics"""
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Apply different threshold strategies
        thresholds = np.arange(0.1, 0.9, 0.1)
        
        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            
            # Business metrics
            approval_rate_actual = (tp + fp) / len(self.y_test)
            default_rate_approved = fp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Financial impact
            total_approved = (tp + fp) * avg_loan_amount
            expected_loss = fp * avg_loan_amount * loss_given_default
            expected_profit = tp * avg_loan_amount * profit_margin
            net_profit = expected_profit - expected_loss
            
            results.append({
                'threshold': threshold,
                'approval_rate': approval_rate_actual,
                'default_rate': default_rate_approved,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'net_profit': net_profit,
                'expected_loss': expected_loss,
                'expected_profit': expected_profit
            })
        
        return pd.DataFrame(results)
    
    def plot_threshold_analysis(self, results_df):
        """Plot threshold analysis for business decision making"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Approval vs Default Rate
        ax1.plot(results_df['threshold'], results_df['approval_rate'], 
                label='Approval Rate', linewidth=2)
        ax1.plot(results_df['threshold'], results_df['default_rate'], 
                label='Default Rate', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Rate')
        ax1.set_title('Approval vs Default Rate by Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Net Profit
        ax2.plot(results_df['threshold'], results_df['net_profit'], 
                linewidth=2, color='green')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Net Profit ($)')
        ax2.set_title('Net Profit by Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Confusion matrix components
        ax3.plot(results_df['threshold'], results_df['true_positives'], 
                label='True Positives', linewidth=2)
        ax3.plot(results_df['threshold'], results_df['false_positives'], 
                label='False Positives', linewidth=2)
        ax3.plot(results_df['threshold'], results_df['false_negatives'], 
                label='False Negatives', linewidth=2)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Components by Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Profit vs Loss
        ax4.plot(results_df['threshold'], results_df['expected_profit'], 
                label='Expected Profit', linewidth=2, color='blue')
        ax4.plot(results_df['threshold'], results_df['expected_loss'], 
                label='Expected Loss', linewidth=2, color='red')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Amount ($)')
        ax4.set_title('Expected Profit vs Loss by Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal threshold (maximizing net profit)
        optimal_idx = results_df['net_profit'].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        
        print(f"Optimal Threshold: {optimal_threshold:.2f}")
        print(f"Maximum Net Profit: ${results_df.loc[optimal_idx, 'net_profit']:,.2f}")
        print(f"Approval Rate at Optimal: {results_df.loc[optimal_idx, 'approval_rate']:.2%}")
        print(f"Default Rate at Optimal: {results_df.loc[optimal_idx, 'default_rate']:.2%}")
        
        return optimal_threshold

    def generate_business_report(self, results_df, optimal_threshold):
        """Generate comprehensive business report"""
        optimal_row = results_df[results_df['threshold'] == optimal_threshold].iloc[0]
        
        report = {
            'optimal_threshold': optimal_threshold,
            'max_net_profit': optimal_row['net_profit'],
            'approval_rate': optimal_row['approval_rate'],
            'default_rate': optimal_row['default_rate'],
            'expected_profit': optimal_row['expected_profit'],
            'expected_loss': optimal_row['expected_loss']
        }
        
        return report