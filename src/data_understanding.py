import pandas as pd
import numpy as np
import io
from pathlib import Path


def generate_data_quality_report(df: pd.DataFrame, target_col: str = "target") -> str:
    """Generate a comprehensive data quality report as a string.
    
    Args:
        df (pd.DataFrame): The dataset to analyze.
        target_col (str): The name of the target variable column. Defaults to 'target'.
        
    Returns:
        str: The formatted data quality report string.
    """
    buffer = io.StringIO()
    
    def write_section(title: str, content: str = "") -> None:
        """Helper function to write formatted sections into the buffer."""
        buffer.write(f"\n{'-'*60}\n")
        buffer.write(f"--- {title} ---\n")
        if content:
            buffer.write(f"{content}\n")
            
    buffer.write("="*60 + "\n")
    buffer.write("           DATASET QUALITY ASSESSMENT REPORT          \n")
    buffer.write("="*60 + "\n")
    
    # 1. Dataset Shape & Dimensions
    write_section("1. Dataset Shape & Dimensions")
    buffer.write(f"Shape            : {df.shape}\n")
    buffer.write(f"Number of Rows   : {df.shape[0]}\n")
    buffer.write(f"Number of Columns: {df.shape[1]}\n")
    
    # 2. First 5 Rows
    write_section("2. First 5 Rows")
    buffer.write(f"{df.head().to_string()}\n")
    
    # 3. Last 5 Rows
    write_section("3. Last 5 Rows")
    buffer.write(f"{df.tail().to_string()}\n")
    
    # 4. Column Names
    write_section("4. Column Names")
    buffer.write(f"{', '.join(df.columns.tolist())}\n")
    
    # 5. Data Types
    write_section("5. Data Types")
    buffer.write(f"{df.dtypes.to_string()}\n")
    
    # 5.1 Data Type Summary
    write_section("5.1 Data Type Summary")
    num_cols = df.select_dtypes(include=['number']).shape[1]
    bool_cols = df.select_dtypes(include=['bool']).shape[1]
    cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    other_cols = df.shape[1] - (num_cols + bool_cols + cat_cols)
    buffer.write(f"Numeric Columns    : {num_cols}\n")
    buffer.write(f"Categorical Columns: {cat_cols}\n")
    buffer.write(f"Boolean Columns    : {bool_cols}\n")
    buffer.write(f"Other Columns      : {other_cols}\n")
    
    # 5.2 Dataset Memory Usage
    write_section("5.2 Dataset Memory Usage")
    mem_usage = df.memory_usage(deep=True)
    buffer.write(f"{mem_usage.to_string()}\n")
    buffer.write(f"\nTotal Memory Usage: {mem_usage.sum() / (1024 ** 2):.4f} MB\n")
    
    # 6. Missing Values Analysis
    write_section("6. Missing Values Analysis")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    missing_total_pct = missing_counts.sum() / (df.shape[0] * df.shape[1]) * 100 if df.shape[0] > 0 else 0
    missing_df = pd.DataFrame({
        "Missing Count": missing_counts, 
        "Missing Percentage (%)": missing_pct
    })
    buffer.write(f"{missing_df.to_string()}\n")
    
    # 7. Duplicate Rows Analysis
    write_section("7. Duplicate Rows Analysis")
    duplicates = df.duplicated().sum()
    duplicate_pct = (duplicates / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    buffer.write(f"Total Duplicate Rows: {duplicates}\n")
    buffer.write(f"Duplicate Percentage: {duplicate_pct:.4f}%\n")
    
    # 8. Unique Values per Column
    write_section("8. Unique Values per Column")
    buffer.write(f"{df.nunique().to_string()}\n")
    
    # 8.1 Constant Columns Detection
    write_section("8.1 Constant Columns Detection")
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        buffer.write(f"Found {len(constant_cols)} constant column(s):\n")
        buffer.write(f"{', '.join(constant_cols)}\n")
    else:
        buffer.write("No constant columns detected.\n")
        
    # 8.2 High Cardinality Columns
    write_section("8.2 High Cardinality Categorical Columns (>50 unique values)")
    cat_cols_list = df.select_dtypes(include=['object', 'category', 'string']).columns
    high_card_cols = [col for col in cat_cols_list if df[col].nunique() > 50]
    if high_card_cols:
        buffer.write(f"Found {len(high_card_cols)} high cardinality categorical column(s):\n")
        buffer.write(f"{', '.join(high_card_cols)}\n")
    else:
        buffer.write("No high cardinality categorical columns detected.\n")

    # 8.3 Low Variance Columns Detection
    write_section("8.3 Low Variance Numeric Columns Detection (Variance < 0.01)")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        low_var_cols = []
        for col in numeric_df.columns:
            if numeric_df[col].var() < 0.01:
                low_var_cols.append(col)
        if low_var_cols:
            buffer.write(f"Found {len(low_var_cols)} numeric column(s) with very low variance:\n")
            buffer.write(f"{', '.join(low_var_cols)}\n")
        else:
            buffer.write("No low variance numeric columns detected.\n")
    else:
        buffer.write("No numeric columns available for variance calculation.\n")

    # 8.4 Outlier Summary using IQR
    write_section("8.4 Outlier Summary (using IQR Method)")
    if not numeric_df.empty:
        outlier_summary = []
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
            if not outliers.empty:
                outlier_summary.append(f"{col}: {len(outliers)} outliers ({(len(outliers)/df.shape[0])*100:.2f}%)")
        
        if outlier_summary:
            buffer.write("\n".join(outlier_summary) + "\n")
        else:
            buffer.write("No outliers detected using the IQR method.\n")
    else:
        buffer.write("No numeric columns available for outlier detection.\n")

    # 8.5 Correlation Summary
    write_section("8.5 Correlation Summary (Top 10 Highest Absolute Correlations)")
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        # Extract the upper triangle to avoid duplicates and self-correlations
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        sorted_corr = upper_tri.unstack().dropna().sort_values(ascending=False)
        top_10 = sorted_corr.head(10)
        
        if not top_10.empty:
            for (col1, col2), val in top_10.items():
                buffer.write(f"{col1} - {col2}: {val:.4f}\n")
        else:
            buffer.write("No valid correlations found.\n")
    else:
        buffer.write("Not enough numeric columns for correlation analysis.\n")

    # 9. Statistical Summary
    write_section("9. Statistical Summary (Describe)")
    buffer.write(f"{df.describe().to_string()}\n")
    
    # 10. Target Column Distribution & Balance
    write_section("10. Target Column Distribution & Balance")
    target_penalty = 0
    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        target_pct = (target_counts / df.shape[0]) * 100
        target_df = pd.DataFrame({
            "Count": target_counts,
            "Class Balance (%)": target_pct
        })
        buffer.write(f"{target_df.to_string()}\n")
        
        # Calculate penalty for Health Score
        majority_class_pct = target_pct.iloc[0] if not target_pct.empty else 0
        if majority_class_pct > 70:
            target_penalty = min(15, (majority_class_pct - 70) * 0.5)
    else:
        buffer.write(f"[NOTE] Target column '{target_col}' not found in the dataset.\n")
        buffer.write(f"       Available columns are: {', '.join(df.columns)}\n")
        
    # 11. Dataset Health Score
    write_section("11. Dataset Health Score (out of 100)")
    health_score = 100.0
    
    # Missing values penalty
    missing_penalty = min(20, missing_total_pct * 2)
    health_score -= missing_penalty
    
    # Duplicates penalty
    duplicate_penalty = min(10, duplicate_pct * 2)
    health_score -= duplicate_penalty
    
    # Constant columns penalty
    constant_penalty = min(15, len(constant_cols) * 5)
    health_score -= constant_penalty
    
    # Target imbalance penalty (calculated in section 10)
    health_score -= target_penalty
    
    health_score = max(0.0, health_score) # Ensure score doesn't go below 0
    
    buffer.write(f"Overall Health Score: {health_score:.2f} / 100\n\n")
    buffer.write("Deductions Breakdown:\n")
    buffer.write(f"- Missing Values   : -{missing_penalty:.2f} pts\n")
    buffer.write(f"- Duplicate Rows   : -{duplicate_penalty:.2f} pts\n")
    buffer.write(f"- Constant Columns : -{constant_penalty:.2f} pts\n")
    if target_col in df.columns:
        buffer.write(f"- Target Imbalance : -{target_penalty:.2f} pts\n")
    
    buffer.write("\n" + "="*60 + "\n")
    buffer.write("          END OF DATASET QUALITY REPORT               \n")
    buffer.write("="*60 + "\n")
    
    return buffer.getvalue()


def save_report_to_file(report: str, file_path: Path) -> None:
    """Save the generated report string to a text file.
    
    Args:
        report (str): The report content to save.
        file_path (Path): The path where the report should be saved.
    """
    try:
        # Ensure the parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[INFO] Data quality report successfully saved to: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save the report to {file_path}: {e}")


def understand_dataset(df: pd.DataFrame, target_col: str = "target") -> None:
    """Perform exploratory data quality assessment and save the report.
    
    This function coordinates generating the report, printing it to the console,
    and saving it to the designated reports directory without modifying the data.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        target_col (str): The name of the target variable column. Defaults to 'target'.
    """
    # 1. Generate the comprehensive report
    report = generate_data_quality_report(df, target_col=target_col)
    
    # 2. Display the report on the console
    print(report)
    
    # 3. Define the output path for the report
    report_path = Path("reports") / "data_quality_report.txt"
    
    # 4. Save the report to the file
    save_report_to_file(report, report_path)
