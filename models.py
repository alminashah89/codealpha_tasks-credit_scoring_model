import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import MODEL_CONFIG, HYPERPARAM_CONFIG

class CreditScoringModels:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def train_models(self):
        """Train multiple classification models"""
        
        print("=== Training Credit Scoring Models ===")
        
        # Define models with initial parameters
        model_configs = {
            'Logistic Regression': LogisticRegression(**MODEL_CONFIG['logistic_regression']),
            'Decision Tree': DecisionTreeClassifier(**MODEL_CONFIG['decision_tree']),
            'Random Forest': RandomForestClassifier(**MODEL_CONFIG['random_forest']),
            'Gradient Boosting': GradientBoostingClassifier(**MODEL_CONFIG['gradient_boosting'])
        }
        
        # Train each model
        for name, model in model_configs.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            self.predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all models using multiple metrics"""
        
        print("\n=== Model Evaluation ===")
        
        metrics_df = pd.DataFrame()
        
        for name in self.models.keys():
            y_pred = self.predictions[name]['y_pred']
            y_pred_proba = self.predictions[name]['y_pred_proba']
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store metrics
            self.metrics[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            }
            
            # Add to DataFrame
            metrics_df[name] = pd.Series({
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
        
        # Display metrics comparison
        print("\nModel Performance Comparison:")
        print(metrics_df.round(4))
        
        return metrics_df
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        
        print("\n=== Hyperparameter Tuning ===")
        
        # Tune Random Forest
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, HYPERPARAM_CONFIG['random_forest'], 
                                 cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train with best parameters
        best_rf = grid_search.best_estimator_
        best_rf.fit(self.X_train, self.y_train)
        
        # Update models and predictions
        self.models['Random Forest (Tuned)'] = best_rf
        y_pred_proba = best_rf.predict_proba(self.X_test)[:, 1]
        self.predictions['Random Forest (Tuned)'] = {
            'y_pred': best_rf.predict(self.X_test),
            'y_pred_proba': y_pred_proba
        }
        
        return best_rf

    def get_best_model(self):
        """Get the best performing model based on ROC-AUC"""
        best_model_name = max(self.metrics.items(), key=lambda x: x[1]['ROC-AUC'])[0]
        return self.models[best_model_name], best_model_name