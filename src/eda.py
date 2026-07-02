"""Module for Exploratory Data Analysis (EDA)."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Output directories for EDA artifacts
REPORTS_DIR = Path("reports") / "eda"
IMAGES_DIR = REPORTS_DIR / "images"

# Automatically create required folders if they don't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Helper function to save figures in high resolution.
    
    Args:
        fig (plt.Figure): The matplotlib figure object.
        filename (str): The name of the file to save (including extension).
    """
    try:
        file_path = IMAGES_DIR / filename
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure: {file_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save figure {filename}: {e}")
    finally:
        # Close the figure to free up memory
        plt.close(fig)


def plot_numerical_distributions(df: pd.DataFrame, num_cols: List[str]) -> None:
    """Generate and save histograms for numerical features.
    
    Args:
        df (pd.DataFrame): The read-only dataset.
        num_cols (List[str]): List of numerical column names.
    """
    logger.info("Generating numerical feature distributions (Histograms).")
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=col, kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        _save_figure(fig, f"histogram_{col}.png")


def plot_boxplots(df: pd.DataFrame, num_cols: List[str]) -> None:
    """Generate and save boxplots for numerical features to inspect outliers.
    
    Args:
        df (pd.DataFrame): The read-only dataset.
        num_cols (List[str]): List of numerical column names.
    """
    logger.info("Generating boxplots for outlier inspection.")
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=col, ax=ax, color='lightgreen')
        ax.set_title(f"Boxplot of {col}", fontsize=14, fontweight='bold')
        ax.set_xlabel(col, fontsize=12)
        _save_figure(fig, f"boxplot_{col}.png")


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: List[str]) -> None:
    """Generate and save a correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): The read-only dataset.
        num_cols (List[str]): List of numerical column names.
    """
    if len(num_cols) < 2:
        logger.warning("Not enough numerical columns to generate correlation heatmap.")
        return
        
    logger.info("Generating correlation heatmap.")
    corr_matrix = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        square=True, 
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=16, fontweight='bold')
    _save_figure(fig, "correlation_heatmap.png")


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """Generate and save the target class distribution plot.
    
    Args:
        df (pd.DataFrame): The read-only dataset.
        target_col (str): The name of the target column.
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found. Skipping target distribution plot.")
        return
        
    logger.info("Generating target class distribution plot.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=target_col, ax=ax, palette='viridis')
    ax.set_title(f"Distribution of Target Variable: {target_col}", fontsize=14, fontweight='bold')
    ax.set_xlabel(target_col, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    # Add counts on top of the bars for clarity
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), 
                    textcoords='offset points')
                    
    _save_figure(fig, "target_distribution.png")


def generate_eda_report(df: pd.DataFrame, target_col: str = "target") -> None:
    """Generate a comprehensive textual EDA report.
    
    Args:
        df (pd.DataFrame): The read-only dataset.
        target_col (str): The name of the target column. Defaults to "target".
    """
    logger.info("Generating textual EDA report.")
    report_path = REPORTS_DIR / "eda_report.txt"
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("          EXPLORATORY DATA ANALYSIS REPORT        \n")
            f.write("="*50 + "\n\n")
            
            # 1. Dataset overview
            f.write("--- 1. Dataset Overview ---\n")
            f.write(f"Total Rows: {df.shape[0]}\n")
            f.write(f"Total Columns: {df.shape[1]}\n\n")
            
            # 2. Missing value analysis
            f.write("--- 2. Missing Value Analysis ---\n")
            missing_stats = df.isnull().sum()
            missing_stats = missing_stats[missing_stats > 0]
            if not missing_stats.empty:
                f.write(missing_stats.to_string() + "\n\n")
            else:
                f.write("No missing values found in the dataset.\n\n")
                
            # 3. Feature summary table
            f.write("--- 3. Feature Summary Table ---\n")
            feature_types = pd.DataFrame({"Data Type": df.dtypes})
            f.write(feature_types.to_string() + "\n\n")
            
            # 4. Basic statistical summary
            f.write("--- 4. Basic Statistical Summary ---\n")
            f.write(df.describe().to_string() + "\n\n")
            
            # 5. Target class distribution
            f.write("--- 5. Target Class Distribution ---\n")
            if target_col in df.columns:
                target_dist = df[target_col].value_counts()
                f.write(target_dist.to_string() + "\n\n")
            else:
                f.write(f"Target column '{target_col}' not found.\n\n")
                
            # 6. Correlation with target
            f.write("--- 6. Correlation with Target ---\n")
            if target_col in df.columns and target_col in num_cols:
                correlations = df[num_cols].corr()[target_col].sort_values(ascending=False)
                # Drop the target itself from the output
                correlations = correlations.drop(target_col)
                f.write(correlations.to_string() + "\n\n")
            else:
                f.write("Cannot compute correlation with target (not found or not numerical).\n\n")
                
        logger.info(f"EDA report successfully saved to {report_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to generate EDA report: {e}")


def perform_eda(df: pd.DataFrame, target_col: str = "target") -> None:
    """Execute the complete Exploratory Data Analysis workflow.
    
    This function coordinates generating textual reports and visualizations
    while strictly keeping the dataset read-only.
    
    Args:
        df (pd.DataFrame): The dataset to analyze.
        target_col (str): The name of the target column. Defaults to "target".
    """
    logger.info("Starting complete Exploratory Data Analysis workflow.")
    
    if df.empty:
        logger.error("Provided dataset is empty. Aborting EDA.")
        return
        
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # 1. Generate text report
    generate_eda_report(df, target_col)
    
    # 2. Generate visualizations
    plot_numerical_distributions(df, num_cols)
    plot_boxplots(df, num_cols)
    plot_correlation_heatmap(df, num_cols)
    plot_target_distribution(df, target_col)
    
    logger.info("Exploratory Data Analysis completed successfully.")
