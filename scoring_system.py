import pandas as pd
import numpy as np

class CreditScoringSystem:
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
    
    def predict_credit_risk(self, applicant_data):
        """Predict credit risk for a new applicant"""
        
        # Convert to DataFrame if single sample
        if isinstance(applicant_data, dict):
            applicant_df = pd.DataFrame([applicant_data])
        else:
            applicant_df = applicant_data.copy()
        
        # Preprocess the data
        categorical_cols = applicant_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in self.preprocessor.label_encoders:
                # Handle unseen categories
                le = self.preprocessor.label_encoders[col]
                applicant_df[col] = applicant_df[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                applicant_df[col] = le.transform(applicant_df[col])
        
        # Impute missing values
        applicant_imputed = pd.DataFrame(
            self.preprocessor.imputer.transform(applicant_df),
            columns=applicant_df.columns
        )
        
        # Scale numerical features
        numerical_cols = applicant_df.select_dtypes(include=[np.number]).columns
        applicant_scaled = applicant_imputed.copy()
        applicant_scaled[numerical_cols] = self.preprocessor.scaler.transform(
            applicant_imputed[numerical_cols]
        )
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in applicant_scaled.columns:
                applicant_scaled[feature] = 0
        
        applicant_scaled = applicant_scaled[self.feature_names]
        
        # Make prediction
        probability = self.model.predict_proba(applicant_scaled)[0, 1]
        prediction = self.model.predict(applicant_scaled)[0]
        
        # Convert to credit score (0-1000 scale)
        credit_score = int((1 - probability) * 1000)
        
        # Risk classification
        if credit_score >= 750:
            risk_category = "Excellent"
        elif credit_score >= 700:
            risk_category = "Good"
        elif credit_score >= 650:
            risk_category = "Fair"
        elif credit_score >= 600:
            risk_category = "Poor"
        else:
            risk_category = "Very Poor"
        
        return {
            'credit_score': credit_score,
            'risk_category': risk_category,
            'default_probability': round(probability, 4),
            'recommendation': 'Approve' if risk_category in ['Excellent', 'Good'] else 'Review' if risk_category == 'Fair' else 'Reject'
        }
    
    def generate_credit_report(self, applicant_data):
        """Generate a comprehensive credit report"""
        
        prediction = self.predict_credit_risk(applicant_data)
        
        print("=== CREDIT SCORING REPORT ===")
        print(f"Credit Score: {prediction['credit_score']}")
        print(f"Risk Category: {prediction['risk_category']}")
        print(f"Default Probability: {prediction['default_probability']:.2%}")
        print(f"Recommendation: {prediction['recommendation']}")
        
        return prediction

    def batch_predict(self, applicants_data):
        """Predict credit risk for multiple applicants"""
        results = []
        for applicant in applicants_data:
            result = self.predict_credit_risk(applicant)
            results.append(result)
        return pd.DataFrame(results)