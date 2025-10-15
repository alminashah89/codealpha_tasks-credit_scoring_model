"""
Main script for Credit Scoring Model
"""

import warnings
warnings.filterwarnings('ignore')

from data_generator import CreditDataGenerator
from data_analyzer import CreditDataAnalyzer
from preprocessor import CreditDataPreprocessor
from models import CreditScoringModels
from evaluator import ModelEvaluator
from scoring_system import CreditScoringSystem
from business_analyzer import BusinessImpactAnalyzer
from utils import save_model_artifacts, create_sample_applicant
from config import BUSINESS_CONFIG

def main():
    print("=== CREDIT SCORING MODEL ===")
    
    # Step 1: Generate or load data
    print("\n1. Generating synthetic credit data...")
    generator = CreditDataGenerator()
    credit_data = generator.generate_synthetic_data()
    generator.save_data(credit_data, 'credit_data.csv')
    
    # Step 2: Exploratory Data Analysis
    print("\n2. Performing exploratory data analysis...")
    analyzer = CreditDataAnalyzer(credit_data)
    correlations = analyzer.perform_eda()
    credit_data_enhanced = analyzer.create_new_features()
    analyzer.plot_target_distribution()
    
    # Step 3: Preprocess data
    print("\n3. Preprocessing data...")
    preprocessor = CreditDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_data(credit_data_enhanced)
    
    # Step 4: Train models
    print("\n4. Training machine learning models...")
    credit_models = CreditScoringModels(X_train, X_test, y_train, y_test)
    credit_models.train_models()
    metrics_df = credit_models.evaluate_models()
    best_model = credit_models.hyperparameter_tuning()
    
    # Step 5: Evaluate models
    print("\n5. Evaluating model performance...")
    evaluator = ModelEvaluator(credit_models.models, credit_models.predictions, X_test, y_test)
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    feature_importance_df = evaluator.feature_importance_analysis(feature_names)
    
    # Step 6: Create scoring system
    print("\n6. Creating credit scoring system...")
    scoring_system = CreditScoringSystem(best_model, preprocessor, feature_names)
    
    # Test with sample applicant
    sample_applicant = create_sample_applicant()
    credit_report = scoring_system.generate_credit_report(sample_applicant)
    
    # Step 7: Business impact analysis
    print("\n7. Analyzing business impact...")
    business_analyzer = BusinessImpactAnalyzer(best_model, X_test, y_test)
    business_results = business_analyzer.calculate_business_metrics(**BUSINESS_CONFIG)
    optimal_threshold = business_analyzer.plot_threshold_analysis(business_results)
    
    # Step 8: Save model artifacts
    print("\n8. Saving model artifacts...")
    save_model_artifacts(best_model, preprocessor, feature_names)
    
    print("\n=== CREDIT SCORING MODEL COMPLETED ===")
    print("Model training and evaluation completed successfully!")
    print(f"Best model: Random Forest (Tuned)")
    print(f"Best ROC-AUC: {metrics_df.loc['ROC-AUC', 'Random Forest (Tuned)']:.4f}")
    print(f"Optimal decision threshold: {optimal_threshold:.2f}")

if __name__ == "__main__":
    main()