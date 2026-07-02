"""Main orchestration script for the Heart Disease Prediction project."""

import logging
from data_loader import load_dataset
from data_understanding import understand_dataset
from eda import perform_eda
from preprocessing import preprocess_dataset
from feature_engineering import perform_feature_engineering
from model_training import (
    load_training_data,
    save_dataset_split_information,
    load_split_datasets,
    train_and_evaluate_logistic_regression,
    train_and_evaluate_svm,
    train_and_evaluate_random_forest,
    train_and_evaluate_xgboost
)
from evaluation import generate_comparative_evaluation
from explainability import generate_shap_summary

# Configure minimal logging for the main orchestrator
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def print_step(step_number: int, step_name: str) -> None:
    """Helper to print clean pipeline progress messages."""
    print(f"\n{'='*40}")
    print(f"STEP {step_number} : {step_name}")
    print(f"{'='*40}\n")


def main() -> None:
    """Execute the main Machine Learning pipeline workflow."""
    
    # Define the primary target column based on the dataset structure
    TARGET_COLUMN = "Target Label"
    
    # ---------------------------------------------------------
    # STEP 1 : DATA LOADING
    # ---------------------------------------------------------
    print_step(1, "DATA LOADING")
    df = load_dataset()
    
    if df is None or df.empty:
        logger.error("Failed to load dataset. Aborting pipeline.")
        return
        
    # Dynamically verify target column existence
    if TARGET_COLUMN not in df.columns:
        if "target" in df.columns:
            TARGET_COLUMN = "target"
        else:
            logger.warning(f"Target column '{TARGET_COLUMN}' not found. Please verify the dataset structure.")
    
    # ---------------------------------------------------------
    # STEP 2 : DATA UNDERSTANDING
    # ---------------------------------------------------------
    print_step(2, "DATA UNDERSTANDING")
    # Generates reports/data_quality_report.txt and outputs health scores
    # We pass the target column for class balance checks if supported by the module
    try:
        understand_dataset(df)
    except TypeError:
        # Fallback if understand_dataset requires a target_col argument
        understand_dataset(df, target_col=TARGET_COLUMN)
    
    # ---------------------------------------------------------
    # STEP 3 : EDA
    # ---------------------------------------------------------
    print_step(3, "EDA")
    # Generates textual report and high-res visualizations in reports/eda/
    perform_eda(df, target_col=TARGET_COLUMN)
    
    # ---------------------------------------------------------
    # STEP 4 : PREPROCESSING
    # ---------------------------------------------------------
    print_step(4, "PREPROCESSING")
    # Handles missing values, scaling, and one-hot encoding
    df_preprocessed, preprocessor = preprocess_dataset(df, target_col=TARGET_COLUMN)
    
    # ---------------------------------------------------------
    # STEP 5 : FEATURE ENGINEERING
    # ---------------------------------------------------------
    print_step(5, "FEATURE ENGINEERING")
    # Handles variance thresholding, correlations, VIF, and feature selection
    # Automatically saves final_features.csv and validation reports
    df_final = perform_feature_engineering(df_preprocessed, target_col=TARGET_COLUMN)
    
    # ---------------------------------------------------------
    # STEP 6 : MODEL TRAINING
    # ---------------------------------------------------------
    print_step(6, "MODEL TRAINING")
    
    import os
    if not os.path.exists(os.path.join("Data", "processed", "X_train.csv")):
        X_tr, X_te, y_tr, y_te = load_training_data()
        save_dataset_split_information(X_tr, X_te, y_tr, y_te)
        
    X_train, X_test, y_train, y_test = load_split_datasets()
    
    try:
        train_and_evaluate_logistic_regression(
            X_train,
            X_test,
            y_train,
            y_test
        )
    except TypeError:
        train_and_evaluate_logistic_regression()
        
    try:
        train_and_evaluate_svm(
            X_train,
            X_test,
            y_train,
            y_test
        )
    except TypeError:
        train_and_evaluate_svm()
        
    try:
        train_and_evaluate_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )
    except TypeError:
        train_and_evaluate_random_forest()
        
    try:
        train_and_evaluate_xgboost(
            X_train,
            X_test,
            y_train,
            y_test
        )
    except TypeError:
        train_and_evaluate_xgboost()
        
    print("MODEL TRAINING COMPLETED")
    
    # ---------------------------------------------------------
    # STEP 7 : COMPARATIVE EVALUATION
    # ---------------------------------------------------------
    print_step(7, "COMPARATIVE EVALUATION")
    generate_comparative_evaluation()
    print("COMPARATIVE EVALUATION COMPLETED")
    
    # ---------------------------------------------------------
    # STEP 8 : MODEL EXPLAINABILITY
    # ---------------------------------------------------------
    print_step(8, "MODEL EXPLAINABILITY")
    generate_shap_summary()
    print("MODEL EXPLAINABILITY COMPLETED")
    
    # ---------------------------------------------------------
    # PIPELINE COMPLETED
    # ---------------------------------------------------------
    print(f"\n{'='*40}")
    print("PIPELINE COMPLETED")
    print(f"{'='*40}\n")
    print(f"Final Processed Dataset Shape: {df_final.shape}")
    print("All research artifacts and validation reports have been successfully generated.")


if __name__ == "__main__":
    main()
