"""Module for model explainability and interpretability using SHAP."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import joblib

try:
    import shap
except ImportError as e:
    raise ImportError(
        "SHAP library is not installed. Please install it using 'pip install shap' "
        "to utilize the explainability framework."
    ) from e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
REPORTS_DIR = Path("reports")
EXPLAINABILITY_DIR = REPORTS_DIR / "explainability"
EXPLAINABILITY_IMAGES_DIR = EXPLAINABILITY_DIR / "images"
MODELS_DIR = Path("models")
PROCESSED_DATA_DIR = Path("Data") / "processed"

# Ensure directories exist
EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
EXPLAINABILITY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def load_saved_models() -> Dict[str, Any]:
    """
    Load all trained research models from the models directory.
    
    Returns:
        Dict[str, Any]: A dictionary mapping model names to their loaded objects.
        
    Raises:
        FileNotFoundError: If a required model file is missing.
        Exception: If there is an issue loading a model using joblib.
    """
    logger.info("Loading saved models for explainability analysis.")
    models = {}
    
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Support Vector Machine": "svm_model.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    for model_name, filename in model_files.items():
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Missing required model file: {filepath}")
            
        try:
            models[model_name] = joblib.load(filepath)
            logger.info(f"Successfully loaded {model_name}.")
        except Exception as e:
            logger.error(f"Failed to load {model_name} from {filepath}: {e}")
            raise
            
    return models


def load_test_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed test dataset for SHAP evaluation.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: X_test features and y_test labels.
        
    Raises:
        FileNotFoundError: If X_test.csv or y_test.csv is missing.
        Exception: If there is an issue reading the CSV files.
    """
    logger.info("Loading test datasets for explainability evaluation.")
    x_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    
    if not x_test_path.exists() or not y_test_path.exists():
        logger.error("Test dataset files (X_test.csv, y_test.csv) are missing.")
        raise FileNotFoundError("Missing test dataset files in Data/processed/.")
        
    try:
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")
        logger.info(f"Successfully loaded X_test (shape: {X_test.shape}) and y_test.")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load test datasets: {e}")
        raise


def create_explainer(model: Any, X_train: Optional[pd.DataFrame] = None) -> Any:
    """
    Dynamically create the appropriate SHAP explainer based on the model type.
    
    Args:
        model (Any): The trained machine learning model.
        X_train (Optional[pd.DataFrame]): Training data, required for Linear/Kernel explainers.
        
    Returns:
        Any: An initialized SHAP explainer object.
        
    Raises:
        ValueError: If the model type is unsupported or if X_train is missing when required.
        Exception: If the explainer initialization fails.
    """
    logger.info(f"Creating SHAP explainer for model type: {type(model).__name__}")
    
    try:
        model_type_name = type(model).__name__
        if model_type_name in ["RandomForestClassifier", "XGBClassifier"]:
            logger.info("Initializing TreeExplainer.")
            return shap.TreeExplainer(model)
            
        elif model_type_name == "LogisticRegression":
            if X_train is None:
                logger.error("LinearExplainer requires X_train as background data.")
                raise ValueError("X_train must be provided for LogisticRegression.")
            logger.info("Initializing LinearExplainer.")
            masker = shap.maskers.Independent(data=X_train)
            return shap.LinearExplainer(model, masker=masker)
            
        elif model_type_name == "SVC":
            if X_train is None:
                logger.error("KernelExplainer requires X_train as background data.")
                raise ValueError("X_train must be provided for SVC.")
            logger.info("Initializing KernelExplainer (using kmeans for background summarization).")
            # Summarize background data for performance
            background = shap.kmeans(X_train, 10)
            return shap.KernelExplainer(model.predict_proba, background)
            
        else:
            logger.error(f"Unsupported model type for explainer: {model_type_name}")
            raise ValueError(f"No explainer configured for model type: {model_type_name}")
            
    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        raise


def save_explanation_report(model_name: str, explanation_summary: str) -> None:
    """
    Save the textual explanation summary to the reports directory.
    
    Args:
        model_name (str): Name of the evaluated model.
        explanation_summary (str): The markdown or text summary of the SHAP findings.
        
    Raises:
        Exception: If the file write operation fails.
    """
    report_filename = f"{model_name.lower().replace(' ', '_')}_shap_report.txt"
    report_path = EXPLAINABILITY_DIR / report_filename
    
    logger.info(f"Saving SHAP explanation report for {model_name}.")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"      SHAP EXPLAINABILITY REPORT: {model_name.upper()}      \n")
            f.write("="*60 + "\n\n")
            f.write(explanation_summary)
            f.write("\n")
        logger.info(f"Successfully saved report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save explanation report for {model_name}: {e}")
        raise


def generate_shap_summary() -> None:
    """
    Generate SHAP summary plots and importance reports for all trained models.
    """
    logger.info("Starting SHAP Summary Generation.")
    
    models = load_saved_models()
    X_test, _ = load_test_dataset()
    
    # Linear and Kernel explainers require X_train for background data
    x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    if x_train_path.exists():
        X_train = pd.read_csv(x_train_path)
    else:
        logger.warning("X_train.csv not found; some explainers may fail.")
        X_train = None
        
    report_path = EXPLAINABILITY_DIR / "shap_summary_report.txt"
    
    summary_text = "="*60 + "\n"
    summary_text += "              SHAP SUMMARY EXPLAINABILITY REPORT              \n"
    summary_text += "="*60 + "\n\n"
    
    filename_map = {
        "Logistic Regression": "logistic_summary.png",
        "Support Vector Machine": "svm_summary.png",
        "Random Forest": "random_forest_summary.png",
        "XGBoost": "xgboost_summary.png"
    }
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    for model_name, model in models.items():
        logger.info(f"Generating SHAP summary for {model_name}.")
        try:
            explainer = create_explainer(model, X_train)
            
            logger.info(f"Calculating SHAP values for {model_name}...")
            shap_values = explainer.shap_values(X_test)
            
            # Extract the numpy array of values for the positive class (or main output)
            if isinstance(shap_values, list):
                vals = shap_values[1]
            elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                vals = shap_values.values[:, :, 1]
            elif hasattr(shap_values, "values"):
                vals = shap_values.values
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                vals = shap_values[:, :, 1]
            else:
                vals = shap_values
                
            # Mean absolute SHAP importance
            mean_abs_shap = np.abs(vals).mean(axis=0)
            feature_names = X_test.columns.tolist()
            
            # Sort by importance
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean_Abs_SHAP": mean_abs_shap
            }).sort_values(by="Mean_Abs_SHAP", ascending=False)
            
            # Generate summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(vals, X_test, show=False)
            image_name = filename_map.get(model_name, f"{model_name.lower().replace(' ', '_')}_summary.png")
            plt.tight_layout()
            plt.savefig(EXPLAINABILITY_IMAGES_DIR / image_name, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP summary plot: {image_name}")
            
            # Add to text report
            summary_text += f"--- {model_name} ---\n"
            summary_text += f"Number of Samples: {X_test.shape[0]}\n"
            summary_text += f"Number of Features: {X_test.shape[1]}\n"
            summary_text += "Top 10 Important Features (Mean Absolute SHAP):\n"
            for _, row in importance_df.head(10).iterrows():
                summary_text += f"  {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}\n"
            summary_text += "\n"
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP summary for {model_name}: {e}")
            
    # Save the combined text report
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        logger.info(f"Saved SHAP summary report to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save SHAP summary report: {e}")


if __name__ == "__main__":
    logger.info("Explainability framework initialized.")
    generate_shap_summary()
