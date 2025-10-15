import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG

class CreditDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
    
    def preprocess_data(self, data, target_col='credit_risk'):
        """Preprocess the credit data for modeling"""
        
        print("=== Data Preprocessing ===")
        
        # Separate features and target
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=DATA_CONFIG['test_size'], random_state=DATA_CONFIG['random_state'], stratify=y
        )
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

    def get_preprocessor_info(self):
        """Get information about the preprocessor"""
        return {
            'scaler': type(self.scaler).__name__,
            'imputer': type(self.imputer).__name__,
            'label_encoders': list(self.label_encoders.keys())
        }