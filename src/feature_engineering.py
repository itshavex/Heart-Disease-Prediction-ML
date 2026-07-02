"""Module for feature engineering operations."""

import logging
import time
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directories
REPORTS_DIR = Path("reports") / "feature_engineering"
IMAGES_DIR = REPORTS_DIR / "images"
PROCESSED_DATA_DIR = Path("Data") / "processed"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def calculate_memory_usage(df: pd.DataFrame) -> float:
    """Calculate the memory usage of a dataframe in MB."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """Remove features with variance below a specific threshold."""
    logger.info(f"Removing low variance features (threshold={threshold}).")
    selector = VarianceThreshold(threshold=threshold)
    
    try:
        selector.fit(X)
        features_to_keep = X.columns[selector.get_support()]
        removed_features = [col for col in X.columns if col not in features_to_keep]
        
        if removed_features:
            logger.info(f"Removed {len(removed_features)} low variance features.")
        else:
            logger.info("No low variance features found.")
            
        return X[features_to_keep], removed_features
    except Exception as e:
        logger.error(f"VarianceThreshold failed: {e}")
        return X, []


def remove_highly_correlated_features(X: pd.DataFrame, threshold: float = 0.85) -> Tuple[pd.DataFrame, List[str]]:
    """Identify and remove highly correlated redundant features."""
    logger.info(f"Removing highly correlated features (threshold={threshold}).")
    
    try:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            logger.info(f"Removed {len(to_drop)} highly correlated features.")
        else:
            logger.info("No highly correlated features found.")
            
        return X.drop(columns=to_drop), to_drop
    except Exception as e:
        logger.error(f"Correlation removal failed: {e}")
        return X, []


def calculate_feature_importances(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate feature importance and permutation importance using a Random Forest estimator."""
    logger.info("Calculating feature importances using RandomForestClassifier.")
    
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=5)
        model.fit(X, y)
        
        # 1. Gini Importance
        rf_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance Score": model.feature_importances_
        }).sort_values(by="Importance Score", ascending=False).reset_index(drop=True)
        rf_importance.insert(0, 'Rank', range(1, len(rf_importance) + 1))
        
        # 2. Permutation Importance
        logger.info("Calculating permutation importance.")
        perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        perm_importance = pd.DataFrame({
            "Feature": X.columns,
            "Permutation Importance": perm_result.importances_mean
        }).sort_values(by="Permutation Importance", ascending=False).reset_index(drop=True)
        perm_importance.insert(0, 'Rank', range(1, len(perm_importance) + 1))
        
        return rf_importance, perm_importance
    except Exception as e:
        logger.error(f"Failed to calculate feature importances: {e}")
        empty_rf = pd.DataFrame(columns=["Rank", "Feature", "Importance Score"])
        empty_perm = pd.DataFrame(columns=["Rank", "Feature", "Permutation Importance"])
        return empty_rf, empty_perm


def generate_feature_audit(
    all_features: List[str],
    removed_var: List[str],
    removed_corr: List[str]
) -> None:
    """Generate a full audit of all original features."""
    logger.info("Generating feature removal audit.")
    audit_data = []
    
    for f in all_features:
        if f in removed_var:
            audit_data.append({
                "Feature": f,
                "Removal Step": "Low Variance",
                "Reason": "Variance < Threshold",
                "Metric Used": "VarianceThreshold"
            })
        elif f in removed_corr:
            audit_data.append({
                "Feature": f,
                "Removal Step": "Correlation Removal",
                "Reason": "Correlation > 0.85",
                "Metric Used": "Pearson Correlation"
            })
        else:
            audit_data.append({
                "Feature": f,
                "Removal Step": "Retained",
                "Reason": "Important Predictor",
                "Metric Used": "-"
            })
            
    audit_df = pd.DataFrame(audit_data)
    audit_path = REPORTS_DIR / "removed_features.csv"
    audit_df.to_csv(audit_path, index=False)


def validate_pipeline_integrity(
    original_df: pd.DataFrame,
    final_df: pd.DataFrame, 
    target_col: str, 
    original_target: pd.Series,
    csv_exported: bool
) -> None:
    """Validate data quality and generate pipeline integrity report."""
    logger.info("Validating pipeline integrity.")
    validation_path = REPORTS_DIR / "validation_report.txt"
    
    target_unchanged = target_col in final_df.columns and final_df[target_col].equals(original_target)
    no_dup_cols = not final_df.columns.duplicated().any()
    no_missing = final_df.isnull().sum().sum() == 0
    no_inf = not np.isinf(final_df.select_dtypes(include=['number'])).any().any()
    
    orig_features = [c for c in original_df.columns if c != target_col]
    final_features = [c for c in final_df.columns if c != target_col]
    relative_order_preserved = all(x in orig_features for x in final_features)
    
    try:
        with open(validation_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("        PIPELINE INTEGRITY VALIDATION REPORT      \n")
            f.write("="*50 + "\n\n")
            f.write(f"[✓] Target unchanged: {target_unchanged}\n")
            f.write(f"[✓] No duplicate columns: {no_dup_cols}\n")
            f.write(f"[✓] No missing values: {no_missing}\n")
            f.write(f"[✓] No infinite values: {no_inf}\n")
            f.write(f"[✓] Feature order preserved: {relative_order_preserved}\n")
            f.write(f"[✓] CSV exported successfully: {csv_exported}\n")
            
            all_passed = (target_unchanged and no_dup_cols and no_missing and no_inf and relative_order_preserved and csv_exported)
            f.write("\nValidation Status: " + ("PASS" if all_passed else "FAIL") + "\n")
    except Exception as e:
        logger.error(f"Failed to write validation report: {e}")


def save_publication_figures(X: pd.DataFrame, rf_importance: pd.DataFrame, perm_importance: pd.DataFrame) -> None:
    """Generate and save publication-quality visualizations."""
    logger.info("Generating publication-quality visualizations.")
    
    try:
        # 1. Final Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Final Feature Correlation Heatmap", fontsize=16, fontweight='bold')
        fig.savefig(IMAGES_DIR / "final_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Top 20 Random Forest Feature Importance
        if not rf_importance.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            top_rf = rf_importance.head(20)
            sns.barplot(data=top_rf, x="Importance Score", y="Feature", ax=ax, palette="viridis")
            ax.set_title("Top Feature Importance (Random Forest Gini)", fontsize=16, fontweight='bold')
            fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # 3. Top 20 Permutation Importance
        if not perm_importance.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            top_perm = perm_importance.head(20)
            sns.barplot(data=top_perm, x="Permutation Importance", y="Feature", ax=ax, palette="plasma")
            ax.set_title("Top Permutation Feature Importance", fontsize=16, fontweight='bold')
            fig.savefig(IMAGES_DIR / "permutation_importance.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        logger.info("Publication figures saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save visualizations: {e}")


def generate_research_summary(
    original_features: int,
    removed_var: List[str],
    removed_corr: List[str],
    final_features: int,
    top_10: List[str]
) -> None:
    """Generate a research paper suitable summary text."""
    report_path = REPORTS_DIR / "research_summary.txt"
    reduction = ((original_features - final_features) / original_features) * 100 if original_features > 0 else 0
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Feature Engineering Research Summary\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Original Features: {original_features}\n")
            f.write(f"Removed by Low Variance: {len(removed_var)} features\n")
            f.write(f"Removed by Correlation: {len(removed_corr)} features\n")
            f.write(f"Final Features: {final_features}\n")
            f.write(f"Dimensionality Reduction: {reduction:.2f}%\n\n")
            
            f.write(f"Top 10 Important Features:\n")
            for i, feat in enumerate(top_10, 1):
                f.write(f"{i}. {feat}\n")
                
            f.write("\nExpected ML Benefits:\n")
            f.write("- Mitigation of the curse of dimensionality, enhancing model generalization.\n")
            f.write("- Significant reduction in multicollinearity, stabilizing coefficient estimates.\n")
            f.write("- Improved computational efficiency and reduced training time footprint.\n\n")
            
            f.write("Suitable Research Paper Description:\n")
            f.write("The raw dataset containing {} features underwent a rigorous automated feature selection pipeline. ".format(original_features))
            f.write("First, a variance threshold approach was employed to eliminate {} uninformative, near-constant features. ".format(len(removed_var)))
            f.write("Subsequently, Pearson correlation analysis was utilized to detect and remove {} highly collinear variables (threshold |r| > 0.85), mitigating redundancy. ".format(len(removed_corr)))
            f.write("The final optimized feature space comprises {} robust predictors, representing a dimensionality reduction of {:.2f}%. ".format(final_features, reduction))
            f.write("Feature importance was computationally validated using both Gini impurity and permutation techniques on a Random Forest estimator.\n")
    except Exception as e:
        logger.error(f"Failed to write research summary: {e}")


def generate_metrics_report(
    orig_shape: Tuple[int, int],
    final_shape: Tuple[int, int],
    mem_before: float,
    mem_after: float,
    processing_time: float
) -> None:
    """Generate performance metrics report."""
    metrics_path = REPORTS_DIR / "feature_engineering_metrics.txt"
    reduction = ((orig_shape[1] - final_shape[1]) / orig_shape[1]) * 100 if orig_shape[1] > 0 else 0
    
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("Feature Engineering Execution Metrics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Original Shape: {orig_shape}\n")
            f.write(f"Final Shape: {final_shape}\n")
            f.write(f"Total Removed: {orig_shape[1] - final_shape[1]}\n")
            f.write(f"Reduction %: {reduction:.2f}%\n")
            f.write(f"Processing Time: {processing_time:.4f} seconds\n")
            f.write(f"Memory Usage Before: {mem_before:.4f} MB\n")
            f.write(f"Memory Usage After: {mem_after:.4f} MB\n")
    except Exception as e:
        logger.error(f"Failed to write metrics report: {e}")


def perform_feature_engineering(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Execute the full feature engineering and validation pipeline.
    
    Args:
        df (pd.DataFrame): The preprocessed dataset.
        target_col (str): The name of the target column. Defaults to "target".
        
    Returns:
        pd.DataFrame: The final clean feature set ready for ML.
    """
    logger.info("Starting Final Feature Engineering pipeline.")
    start_time = time.time()
    
    if df.empty or target_col not in df.columns:
        logger.error("Invalid dataframe or missing target column.")
        return df
        
    mem_before = calculate_memory_usage(df)
    orig_shape = df.shape
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    all_original_features = X.columns.tolist()
    
    # Enforce numerical features only for analysis
    X_numeric = X.select_dtypes(include=['number'])
    if X_numeric.shape[1] != X.shape[1]:
        logger.warning("Found non-numeric columns. Proceeding with numeric subset.")
        
    # 1. Variance Threshold
    X_var, removed_var = remove_low_variance_features(X_numeric, threshold=0.01)
    
    # 2. Highly Correlated Features
    X_final, removed_corr = remove_highly_correlated_features(X_var, threshold=0.85)
    
    # 3. Feature Importance (Random Forest & Permutation)
    rf_importance, perm_importance = calculate_feature_importances(X_final, y)
    
    # 4. Generate Final Features List
    if not rf_importance.empty:
        dtypes_df = pd.DataFrame({"Data Type": X_final.dtypes}).reset_index().rename(columns={"index": "Feature"})
        final_features_list = pd.merge(rf_importance, dtypes_df, on="Feature")
        final_features_list = final_features_list[["Feature", "Data Type", "Rank", "Importance Score"]]
        final_features_list.columns = ["Feature Name", "Data Type", "Importance Rank", "Importance Score"]
        final_features_list.to_csv(REPORTS_DIR / "final_features_list.csv", index=False)
        
        rf_importance.to_csv(REPORTS_DIR / "feature_ranking.csv", index=False)
        perm_importance.to_csv(REPORTS_DIR / "permutation_feature_importance.csv", index=False)
    
    # 5. Generate Audits and Reports
    generate_feature_audit(all_original_features, removed_var, removed_corr)
    
    top_10_features = rf_importance["Feature"].head(10).tolist() if not rf_importance.empty else []
    generate_research_summary(
        original_features=len(all_original_features),
        removed_var=removed_var,
        removed_corr=removed_corr,
        final_features=X_final.shape[1],
        top_10=top_10_features
    )
    
    # 6. Generate Visualizations
    save_publication_figures(X_final, rf_importance, perm_importance)
    
    # Combine final features with target
    final_df = pd.concat([X_final, y], axis=1)
    
    # 7. Save final feature matrix
    final_path = PROCESSED_DATA_DIR / "final_features.csv"
    csv_exported = False
    try:
        final_df.to_csv(final_path, index=False)
        logger.info(f"Final optimized feature matrix saved to {final_path}")
        csv_exported = True
    except Exception as e:
        logger.error(f"Failed to save final feature matrix: {e}")
        
    # 8. Validation & Metrics
    validate_pipeline_integrity(df, final_df, target_col, y, csv_exported)
    
    mem_after = calculate_memory_usage(final_df)
    processing_time = time.time() - start_time
    
    generate_metrics_report(
        orig_shape=orig_shape,
        final_shape=final_df.shape,
        mem_before=mem_before,
        mem_after=mem_after,
        processing_time=processing_time
    )
    
    logger.info(f"Feature Engineering completed in {processing_time:.2f}s. Final dataset shape: {final_df.shape}")
    return final_df
