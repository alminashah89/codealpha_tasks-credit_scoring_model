import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

class ModelEvaluator:
    def __init__(self, models, predictions, X_test, y_test):
        self.models = models
        self.predictions = predictions
        self.X_test = X_test
        self.y_test = y_test
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        
        n_models = min(len(self.models), 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (name, pred) in enumerate(list(self.predictions.items())[:4]):
            cm = confusion_matrix(self.y_test, pred['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        
        plt.figure(figsize=(10, 8))
        
        for name, pred in self.predictions.items():
            fpr, tpr, _ = roc_curve(self.y_test, pred['y_pred_proba'])
            auc_score = roc_auc_score(self.y_test, pred['y_pred_proba'])
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Credit Scoring Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self):
        """Plot Precision-Recall curves"""
        
        plt.figure(figsize=(10, 8))
        
        for name, pred in self.predictions.items():
            precision, recall, _ = precision_recall_curve(self.y_test, pred['y_pred_proba'])
            
            plt.plot(recall, precision, label=f'{name}', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Credit Scoring Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def feature_importance_analysis(self, feature_names, top_n=10):
        """Analyze and plot feature importance"""
        
        # Get the best model (Random Forest)
        best_model = self.models.get('Random Forest (Tuned)', 
                                   self.models.get('Random Forest'))
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_imp_df.head(top_n), x='importance', y='feature')
            plt.title(f'Top {top_n} Most Important Features for Credit Scoring')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return feature_imp_df
        
        return None

    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        report = {}
        for name, pred in self.predictions.items():
            y_pred = pred['y_pred']
            y_pred_proba = pred['y_pred_proba']
            
            report[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
        
        return pd.DataFrame(report).T