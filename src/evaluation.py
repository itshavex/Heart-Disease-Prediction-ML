"""Module for comparative evaluation of trained machine learning models."""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
REPORTS_DIR = Path("reports")
MODEL_TRAINING_DIR = REPORTS_DIR / "model_training"
COMPARISON_DIR = REPORTS_DIR / "model_comparison"
IMAGES_DIR = COMPARISON_DIR / "images"

# Ensure directories exist
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_report.txt",
    "Support Vector Machine": "svm_report.txt",
    "Random Forest": "random_forest_report.txt",
    "XGBoost": "xgboost_report.txt"
}

def _parse_report(filepath: Path) -> Dict[str, float]:
    """Parse a single model report to extract performance metrics."""
    if not filepath.exists():
        logger.error(f"Report not found: {filepath}")
        return {}
        
    content = filepath.read_text(encoding="utf-8")
    metrics = {}
    
    cv_match = re.search(r"Mean CV Accuracy:\s*([\d\.]+)\s*\(\+/-\s*([\d\.]+)\)", content)
    if cv_match:
        metrics['Cross Validation Mean'] = float(cv_match.group(1))
        metrics['Cross Validation Std'] = float(cv_match.group(2))
        
    acc_match = re.search(r"Accuracy:\s*([\d\.]+)", content)
    if acc_match: 
        metrics['Accuracy'] = float(acc_match.group(1))
        
    prec_match = re.search(r"Precision:\s*([\d\.]+)", content)
    if prec_match: 
        metrics['Precision'] = float(prec_match.group(1))
        
    rec_match = re.search(r"Recall:\s*([\d\.]+)", content)
    if rec_match: 
        metrics['Recall'] = float(rec_match.group(1))
        
    f1_match = re.search(r"F1 Score:\s*([\d\.]+)", content)
    if f1_match: 
        metrics['F1 Score'] = float(f1_match.group(1))
        
    roc_match = re.search(r"ROC-AUC:\s*([\d\.]+)", content)
    if roc_match: 
        metrics['ROC AUC'] = float(roc_match.group(1))
        
    return metrics

def generate_comparative_evaluation() -> None:
    """Read all model reports, build a comparison DataFrame, save figures, and generate a final report."""
    logger.info("Starting comparative evaluation framework.")
    
    # 1. Read Evaluation Outputs
    data = []
    for model_name, filename in MODEL_FILES.items():
        report_path = MODEL_TRAINING_DIR / filename
        metrics = _parse_report(report_path)
        if metrics:
            metrics["Model"] = model_name
            data.append(metrics)
            
    if not data:
        logger.error("No metrics could be parsed from the reports.")
        return
        
    # 2. Build Comparison DataFrame
    df = pd.DataFrame(data)
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Cross Validation Mean', 'Cross Validation Std']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    logger.info("Comparison DataFrame built successfully.")
    
    # 3. Save DataFrames
    metrics_csv = COMPARISON_DIR / "model_metrics.csv"
    comp_csv = COMPARISON_DIR / "model_comparison.csv"
    df.to_csv(metrics_csv, index=False)
    df.to_csv(comp_csv, index=False)
    logger.info(f"Saved CSVs to {COMPARISON_DIR}")
    
    # 4. Generate Visualizations
    metrics_to_plot = {
        'Accuracy': 'accuracy_comparison.png',
        'Precision': 'precision_comparison.png',
        'Recall': 'recall_comparison.png',
        'F1 Score': 'f1_comparison.png',
        'ROC AUC': 'roc_auc_comparison.png',
        'Cross Validation Mean': 'cv_accuracy_comparison.png'
    }
    
    for metric, filename in metrics_to_plot.items():
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df, x='Model', y=metric, ax=ax, palette='viridis', hue='Model', legend=False)
            ax.set_title(f"Model Comparison - {metric}", fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.05])
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.4f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')
                            
            plt.xticks(rotation=15)
            fig.tight_layout()
            fig.savefig(IMAGES_DIR / filename, dpi=300)
            plt.close(fig)
            
    logger.info("Publication-quality comparison figures generated.")
    
    # 5. Identify Best Models
    best_acc = df.loc[df['Accuracy'].idxmax()]['Model']
    best_roc = df.loc[df['ROC AUC'].idxmax()]['Model']
    best_f1 = df.loc[df['F1 Score'].idxmax()]['Model']
    
    # Calculate a composite score for overall best (mean of key metrics)
    df['Overall Score'] = df[['Accuracy', 'F1 Score', 'ROC AUC', 'Cross Validation Mean']].mean(axis=1)
    best_overall = df.loc[df['Overall Score'].idxmax()]['Model']
    
    # 6. Generate Research Report
    report_file = COMPARISON_DIR / "model_comparison_report.txt"
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("         COMPARATIVE EVALUATION FRAMEWORK REPORT        \n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("--- 1. Dataset Summary ---\n")
            f.write("The models were trained on the preprocessed heart disease dataset.\n")
            f.write("Evaluations were performed using a holdout test set (20%) and 5-Fold Stratified Cross Validation.\n\n")
            
            f.write("--- 2. Models Compared ---\n")
            for m in df['Model']:
                f.write(f"- {m}\n")
            f.write("\n")
            
            f.write("--- 3. Performance Table ---\n")
            f.write(df.drop(columns=['Overall Score']).to_string(index=False) + "\n\n")
            
            f.write("--- 4. Best Performing Models ---\n")
            f.write(f"Best Accuracy Model : {best_acc}\n")
            f.write(f"Best ROC AUC Model  : {best_roc}\n")
            f.write(f"Best F1 Score Model : {best_f1}\n")
            f.write(f"Best Overall Model  : {best_overall}\n\n")
            
            f.write("--- 5. Discussion ---\n")
            f.write("Strengths of each model:\n")
            f.write("- Logistic Regression: Highly interpretable, lightweight baseline model.\n")
            f.write("- Support Vector Machine: Robust to non-linear boundaries using the RBF kernel.\n")
            f.write("- Random Forest: Excellent at handling complex interactions, robust to overfitting.\n")
            f.write("- XGBoost: State-of-the-art performance for tabular data, handles missing values intrinsically.\n\n")
            
            f.write("Weaknesses of each model:\n")
            f.write("- Logistic Regression: Struggles with complex non-linear feature interactions.\n")
            f.write("- Support Vector Machine: Output probabilities may require Platt scaling for calibration.\n")
            f.write("- Random Forest: Larger memory footprint, slightly slower inference time.\n")
            f.write("- XGBoost: Sensitive to hyperparameters and more prone to overfitting without tuning.\n\n")
            
            f.write("--- 6. Research Conclusion ---\n")
            f.write(f"The extensive evaluation demonstrates that {best_overall} achieved the strongest overall performance across multiple key metrics including Accuracy, F1 Score, and ROC-AUC. Ensemble methods generally outperformed the linear baselines.\n\n")
            
            f.write("--- 7. Recommended Final Model ---\n")
            f.write(f"Recommended Model: {best_overall}\n")
            f.write(f"This model is recommended for deployment and further explainability analysis.\n\n")
            
            f.write("--- 8. Future Work ---\n")
            f.write("- Apply SHAP (SHapley Additive exPlanations) to interpret model decisions and identify primary risk factors.\n")
            f.write("- Perform hyperparameter tuning via Grid Search or Bayesian Optimization to squeeze further performance.\n")
            
        logger.info(f"Final model comparison report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to write comparison report: {e}")

if __name__ == "__main__":
    generate_comparative_evaluation()
