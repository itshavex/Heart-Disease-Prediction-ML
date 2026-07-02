"""Module for model training framework and data preparation."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve, auc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Core Directories
PROCESSED_DATA_DIR = Path("Data") / "processed"
REPORTS_DIR = Path("reports") / "model_training"
IMAGES_DIR = REPORTS_DIR / "images"
MODELS_DIR = Path("models")

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# FRAMEWORK: DATASET SPLITTING (From Sprint 6.1)
# ---------------------------------------------------------

def _detect_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the target column in the dataset."""
    possible_targets = ["target", "Target", "target_label", "Target Label"]
    for t in possible_targets:
        if t in df.columns:
            logger.info(f"Detected target column: '{t}'")
            return t
    logger.error(f"Target column not found. Checked for: {possible_targets}")
    raise ValueError("Target column is missing from the dataset.")


def load_training_data(
    filepath: Path = PROCESSED_DATA_DIR / "final_features.csv", 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the final feature dataset and split into training and testing sets."""
    logger.info(f"Loading processed dataset from {filepath}")
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Dataset is empty.")
        
    target_col = _detect_target_column(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Performing train-test split (test_size={test_size}, random_state={random_state}).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_cross_validator(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Create a highly reproducible StratifiedKFold cross-validator."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def save_dataset_split_information(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    random_state: int = 42
) -> None:
    """Save the train/test splits and generate a configuration report."""
    logger.info("Saving training and testing datasets to disk.")
    try:
        X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
        X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
        y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
        y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)
    except Exception as e:
        logger.error(f"Failed to save split datasets: {e}")
        raise

    report_path = REPORTS_DIR / "train_test_split_report.txt"
    total_samples = X_train.shape[0] + X_test.shape[0]
    train_pct = (X_train.shape[0] / total_samples) * 100 if total_samples > 0 else 0
    test_pct = (X_test.shape[0] / total_samples) * 100 if total_samples > 0 else 0
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n             TRAIN-TEST SPLIT REPORT               \n" + "="*60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("--- 1. Dataset Dimensions ---\n")
            f.write(f"Original Dataset Shape: ({total_samples}, {X_train.shape[1] + 1})\n")
            f.write(f"Training Shape: {X_train.shape}\nTesting Shape: {X_test.shape}\n\n")
            f.write("--- 2. Split Proportions ---\n")
            f.write(f"Training Percentage: {train_pct:.2f}%\nTesting Percentage: {test_pct:.2f}%\n\n")
            f.write("--- 3. Target Distribution (y_train) ---\n")
            f.write("Normalized Class Proportions:\n" + y_train.value_counts(normalize=True).to_string() + "\n\n")
            f.write("--- 4. Configuration ---\n")
            f.write(f"Random State: {random_state}\nStratification: Enabled (y parameter)\n")
            f.write("Cross Validation Configuration: StratifiedKFold (n_splits=5, shuffle=True)\n")
    except Exception as e:
        logger.error(f"Failed to generate train-test split report: {e}")
        raise


# ---------------------------------------------------------
# SPRINT 6.2: LOGISTIC REGRESSION TRAINING
# ---------------------------------------------------------

def load_split_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the pre-split training and testing datasets from disk."""
    logger.info("Loading X_train, X_test, y_train, y_test from disk.")
    try:
        X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        # Ensure target is loaded as Series (1D) rather than DataFrame
        y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").iloc[:, 0]
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0]
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to load split datasets: {e}")
        raise


def train_and_evaluate_logistic_regression() -> None:
    """Train Logistic Regression, evaluate, and save artifacts."""
    logger.info("Starting Logistic Regression training and evaluation.")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_split_datasets()
    
    # 2. Initialize Model
    logger.info("Initializing LogisticRegression (random_state=42, max_iter=1000).")
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 3. Cross Validation
    logger.info("Performing 5-Fold Stratified Cross Validation.")
    cv = create_cross_validator(n_splits=5, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    logger.info(f"CV Accuracy Scores: {cv_scores}")
    logger.info(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # 4. Train Final Model
    logger.info("Training final LogisticRegression model on full training set.")
    model.fit(X_train, y_train)
    
    # 5. Predictions & Probabilities
    logger.info("Generating predictions on test set.")
    y_pred = model.predict(X_test)
    
    # Safely extract probabilities for positive class
    if len(model.classes_) == 2:
        y_prob = model.predict_proba(X_test)[:, 1] 
    else:
        logger.warning("Dataset is not strictly binary, using max probability instead of positive class.")
        y_prob = np.max(model.predict_proba(X_test), axis=1)
    
    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }
    
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 7. Save Model
    model_path = MODELS_DIR / "logistic_regression.pkl"
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
        
    # 8. Save Predictions
    try:
        pd.DataFrame({"y_pred": y_pred}).to_csv(PROCESSED_DATA_DIR / "lr_predictions.csv", index=False)
        pd.DataFrame({"y_prob": y_prob}).to_csv(PROCESSED_DATA_DIR / "lr_probabilities.csv", index=False)
        logger.info("Predictions saved to Data/processed/")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        
    # 9. Generate Report
    report_path = REPORTS_DIR / "logistic_regression_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("      LOGISTIC REGRESSION TRAINING REPORT      \n")
            f.write("="*50 + "\n\n")
            
            f.write("--- 1. Cross Validation (5-Fold Stratified) ---\n")
            f.write(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})\n\n")
            
            f.write("--- 2. Test Set Evaluation Metrics ---\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")
            
            f.write("--- 3. Classification Report ---\n")
            f.write(class_report + "\n")
            
            f.write("--- 4. Confusion Matrix ---\n")
            f.write(str(conf_matrix) + "\n")
        logger.info(f"Report saved to {report_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write report: {e}")
        
    # 10. Generate Visualizations
    _save_visualizations(y_test, y_pred, y_prob, conf_matrix)


def _save_visualizations(
    y_test: pd.Series, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    conf_matrix: np.ndarray,
    model_name: str = "Logistic Regression",
    file_suffix: str = "lr"
) -> None:
    """Generate and save publication-quality evaluation figures."""
    logger.info("Generating evaluation figures.")
    
    try:
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{model_name} Confusion Matrix", fontsize=14, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        fig.savefig(IMAGES_DIR / f"confusion_matrix_{file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name} ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        fig.savefig(IMAGES_DIR / f"roc_curve_{file_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info("Visualizations saved to images directory.")
    except Exception as e:
        logger.error(f"Failed to generate evaluation figures: {e}")


# ---------------------------------------------------------
# SPRINT 6.3: SUPPORT VECTOR MACHINE TRAINING
# ---------------------------------------------------------

def train_and_evaluate_svm() -> Dict[str, Any]:
    """Train Support Vector Machine, evaluate, save artifacts, and return metrics."""
    logger.info("Starting Support Vector Machine training and evaluation.")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_split_datasets()
    
    # 2. Initialize Model
    logger.info("Initializing SVC (kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42).")
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    
    # 3. Cross Validation
    logger.info("Performing 5-Fold Stratified Cross Validation.")
    cv = create_cross_validator(n_splits=5, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info(f"CV Accuracy Scores: {cv_scores}")
    logger.info(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # 4. Train Final Model
    logger.info("Training final SVC model on full training set.")
    model.fit(X_train, y_train)
    
    # 5. Predictions & Probabilities
    logger.info("Generating predictions on test set.")
    y_pred = model.predict(X_test)
    
    # Safely extract probabilities for positive class
    if len(model.classes_) == 2:
        y_prob = model.predict_proba(X_test)[:, 1] 
    else:
        logger.warning("Dataset is not strictly binary, using max probability instead of positive class.")
        y_prob = np.max(model.predict_proba(X_test), axis=1)
    
    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }
    
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 7. Save Model
    model_path = MODELS_DIR / "svm_model.pkl"
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
        
    # 8. Save Predictions
    try:
        pd.DataFrame({"y_pred": y_pred}).to_csv(PROCESSED_DATA_DIR / "svm_predictions.csv", index=False)
        pd.DataFrame({"y_prob": y_prob}).to_csv(PROCESSED_DATA_DIR / "svm_probabilities.csv", index=False)
        logger.info("Predictions saved to Data/processed/")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        
    # 9. Generate Report
    report_path = REPORTS_DIR / "svm_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("      SUPPORT VECTOR MACHINE TRAINING REPORT      \n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- 1. Dataset Shape ---\n")
            f.write(f"Training Shape: {X_train.shape}\nTesting Shape: {X_test.shape}\n\n")
            
            f.write("--- 2. Model Configuration ---\n")
            f.write("Kernel: rbf\nC: 1.0\nGamma: scale\nProbability: True\nRandom State: 42\n\n")
            
            f.write("--- 3. Cross Validation (5-Fold Stratified) ---\n")
            f.write(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})\n\n")
            
            f.write("--- 4. Test Set Evaluation Metrics ---\n")
            for k, v in metrics_dict.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")
            
            f.write("--- 5. Classification Report ---\n")
            f.write(class_report + "\n")
            
            f.write("--- 6. Confusion Matrix ---\n")
            f.write(str(conf_matrix) + "\n")
        logger.info(f"Report saved to {report_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write report: {e}")
        
    # 10. Generate Visualizations
    _save_visualizations(y_test, y_pred, y_prob, conf_matrix, model_name="Support Vector Machine", file_suffix="svm")

    # 11. Return Metrics
    return {
        "model": "Support Vector Machine",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "cv_mean_accuracy": float(cv_mean),
        "cv_std_accuracy": float(cv_std)
    }


# ---------------------------------------------------------
# SPRINT 6.4: RANDOM FOREST TRAINING
# ---------------------------------------------------------

def train_and_evaluate_random_forest() -> Dict[str, Any]:
    """Train Random Forest, evaluate, save artifacts, and return metrics."""
    logger.info("Starting Random Forest training and evaluation.")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_split_datasets()
    
    # 2. Initialize Model
    logger.info("Initializing RandomForestClassifier (n_estimators=200, random_state=42, n_jobs=-1).")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    # 3. Cross Validation
    logger.info("Performing 5-Fold Stratified Cross Validation.")
    cv = create_cross_validator(n_splits=5, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info(f"CV Accuracy Scores: {cv_scores}")
    logger.info(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # 4. Train Final Model
    logger.info("Training final Random Forest model on full training set.")
    model.fit(X_train, y_train)
    
    # 5. Predictions & Probabilities
    logger.info("Generating predictions on test set.")
    y_pred = model.predict(X_test)
    
    # Safely extract probabilities for positive class
    if len(model.classes_) == 2:
        y_prob = model.predict_proba(X_test)[:, 1] 
    else:
        logger.warning("Dataset is not strictly binary, using max probability instead of positive class.")
        y_prob = np.max(model.predict_proba(X_test), axis=1)
    
    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }
    
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 7. Save Model
    model_path = MODELS_DIR / "random_forest.pkl"
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
        
    # 8. Save Predictions
    try:
        pd.DataFrame({"y_pred": y_pred}).to_csv(PROCESSED_DATA_DIR / "random_forest_predictions.csv", index=False)
        pd.DataFrame({"y_prob": y_prob}).to_csv(PROCESSED_DATA_DIR / "random_forest_probabilities.csv", index=False)
        logger.info("Predictions saved to Data/processed/")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        
    # 9. Generate Report
    report_path = REPORTS_DIR / "random_forest_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("      RANDOM FOREST TRAINING REPORT      \n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- 1. Dataset Shape ---\n")
            f.write(f"Training Shape: {X_train.shape}\nTesting Shape: {X_test.shape}\n\n")
            
            f.write("--- 2. Model Configuration ---\n")
            f.write("n_estimators: 200\nrandom_state: 42\nn_jobs: -1\n\n")
            
            f.write("--- 3. Cross Validation (5-Fold Stratified) ---\n")
            f.write(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})\n\n")
            
            f.write("--- 4. Test Set Evaluation Metrics ---\n")
            for k, v in metrics_dict.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")
            
            f.write("--- 5. Classification Report ---\n")
            f.write(class_report + "\n")
            
            f.write("--- 6. Confusion Matrix ---\n")
            f.write(str(conf_matrix) + "\n\n")
            
            f.write("--- 7. Research Conclusions ---\n")
            f.write("Random Forest models often provide robust baseline performances with minimal hyperparameter tuning.\n")
            f.write("Future work could involve feature importance analysis and grid search optimization.\n")
            
        logger.info(f"Report saved to {report_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write report: {e}")
        
    # 10. Generate Visualizations
    _save_visualizations(y_test, y_pred, y_prob, conf_matrix, model_name="Random Forest", file_suffix="rf")

    # 11. Return Metrics
    return {
        "model": "Random Forest",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "cv_mean_accuracy": float(cv_mean),
        "cv_std_accuracy": float(cv_std)
    }


# ---------------------------------------------------------
# SPRINT 6.5: XGBOOST TRAINING
# ---------------------------------------------------------

def train_and_evaluate_xgboost() -> Dict[str, Any]:
    """Train XGBoost, evaluate, save artifacts, and return metrics."""
    logger.info("Starting XGBoost training and evaluation.")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_split_datasets()
    
    # 2. Initialize Model
    logger.info("Initializing XGBClassifier (random_state=42, n_jobs=-1, eval_metric='logloss').")
    model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    
    # 3. Cross Validation
    logger.info("Performing 5-Fold Stratified Cross Validation.")
    cv = create_cross_validator(n_splits=5, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info(f"CV Accuracy Scores: {cv_scores}")
    logger.info(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # 4. Train Final Model
    logger.info("Training final XGBoost model on full training set.")
    model.fit(X_train, y_train)
    
    # 5. Predictions & Probabilities
    logger.info("Generating predictions on test set.")
    y_pred = model.predict(X_test)
    
    # Safely extract probabilities for positive class
    if len(model.classes_) == 2:
        y_prob = model.predict_proba(X_test)[:, 1] 
    else:
        logger.warning("Dataset is not strictly binary, using max probability instead of positive class.")
        y_prob = np.max(model.predict_proba(X_test), axis=1)
    
    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }
    
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 7. Save Model
    model_path = MODELS_DIR / "xgboost.pkl"
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
        
    # 8. Save Predictions
    try:
        pd.DataFrame({"y_pred": y_pred}).to_csv(PROCESSED_DATA_DIR / "xgboost_predictions.csv", index=False)
        pd.DataFrame({"y_prob": y_prob}).to_csv(PROCESSED_DATA_DIR / "xgboost_probabilities.csv", index=False)
        logger.info("Predictions saved to Data/processed/")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        
    # 9. Generate Report
    report_path = REPORTS_DIR / "xgboost_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("      XGBOOST TRAINING REPORT      \n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- 1. Dataset Shape ---\n")
            f.write(f"Training Shape: {X_train.shape}\nTesting Shape: {X_test.shape}\n\n")
            
            f.write("--- 2. Model Configuration ---\n")
            f.write("random_state: 42\nn_jobs: -1\neval_metric: logloss\n\n")
            
            f.write("--- 3. Cross Validation (5-Fold Stratified) ---\n")
            f.write(f"Mean CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})\n\n")
            
            f.write("--- 4. Test Set Evaluation Metrics ---\n")
            for k, v in metrics_dict.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")
            
            f.write("--- 5. Classification Report ---\n")
            f.write(class_report + "\n")
            
            f.write("--- 6. Confusion Matrix ---\n")
            f.write(str(conf_matrix) + "\n\n")
            
            f.write("--- 7. Research Conclusions ---\n")
            f.write("XGBoost typically provides state-of-the-art performance for tabular data.\n")
            f.write("Further hyperparameter tuning (e.g., learning rate, max depth) is recommended.\n")
            
        logger.info(f"Report saved to {report_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write report: {e}")
        
    # 10. Generate Visualizations
    _save_visualizations(y_test, y_pred, y_prob, conf_matrix, model_name="XGBoost", file_suffix="xgb")

    # 11. Return Metrics
    return {
        "model": "XGBoost",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "cv_mean_accuracy": float(cv_mean),
        "cv_std_accuracy": float(cv_std)
    }
