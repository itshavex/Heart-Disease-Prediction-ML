"""Module for data preprocessing operations."""

import logging
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configure logging for the preprocessing module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def identify_feature_types(df: pd.DataFrame, target_col: str = "target") -> Tuple[List[str], List[str]]:
    """Automatically identify numerical and categorical features in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze.
        target_col (str): The name of the target column to exclude from feature lists. Defaults to "target".
        
    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of numerical column names 
                                     and a list of categorical column names.
                                     
    Raises:
        ValueError: If the dataframe is empty.
    """
    if df.empty:
        logger.error("The provided dataframe is empty. Cannot identify features.")
        raise ValueError("Dataframe is empty.")

    try:
        logger.info("Identifying feature types in the dataset.")
        
        # Exclude target column from feature identification if it exists
        features_df = df.drop(columns=[target_col]) if target_col in df.columns else df
        
        numerical_cols = features_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = features_df.select_dtypes(exclude=['number']).columns.tolist()
        
        logger.info(f"Identified {len(numerical_cols)} numerical features.")
        logger.info(f"Identified {len(categorical_cols)} categorical features.")
        
        return numerical_cols, categorical_cols
    except Exception as e:
        logger.error(f"Error while identifying feature types: {e}")
        raise


def build_preprocessing_pipeline(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build the scikit-learn preprocessing pipeline.
    
    Args:
        numerical_cols (List[str]): List of numerical column names.
        categorical_cols (List[str]): List of categorical column names.
        
    Returns:
        ColumnTransformer: The configured scikit-learn preprocessing pipeline.
        
    Raises:
        Exception: If pipeline construction fails.
    """
    try:
        logger.info("Building the preprocessing pipeline.")
        
        # 1. Pipeline for numerical features: Impute with median, then scale
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # 2. Pipeline for categorical features: Impute with most frequent, then one-hot encode
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False ensures a dense matrix is returned, avoiding compatibility issues with DataFrames
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        # 3. Combine both pipelines using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
            ],
            remainder="drop" # Drop any columns not explicitly specified (e.g., target)
        )
        
        logger.info("Preprocessing pipeline constructed successfully.")
        return preprocessor
    except Exception as e:
        logger.error(f"Error while building the preprocessing pipeline: {e}")
        raise


def preprocess_dataset(
    df: pd.DataFrame, 
    target_col: str = "target"
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Execute the full data preprocessing workflow.
    
    Args:
        df (pd.DataFrame): The raw dataset to preprocess.
        target_col (str): The name of the target column to isolate. Defaults to "target".
        
    Returns:
        Tuple[pd.DataFrame, ColumnTransformer]: The fully preprocessed dataset (including the target) 
                                                and the fitted preprocessing pipeline.
                                                
    Raises:
        Exception: If any step in the preprocessing workflow fails.
    """
    if df.empty:
        logger.error("The provided dataframe is empty. Preprocessing aborted.")
        raise ValueError("Dataframe is empty.")

    try:
        logger.info("Starting dataset preprocessing workflow.")
        
        # Work on a copy to avoid unintended modifications to the original dataframe
        data = df.copy()
        
        # Isolate the target variable before transformation to preserve it
        y = None
        if target_col in data.columns:
            y = data[target_col]
            X = data.drop(columns=[target_col])
            logger.info(f"Target column '{target_col}' isolated successfully.")
        else:
            logger.warning(f"Target column '{target_col}' not found. Preprocessing all columns.")
            X = data
            
        # 1. Identify Feature Types
        numerical_cols, categorical_cols = identify_feature_types(X, target_col=target_col)
        
        # 2. Build Pipeline
        preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
        
        # 3. Fit and Transform
        logger.info("Fitting the pipeline and transforming the features.")
        X_transformed = preprocessor.fit_transform(X)
        
        # Extract feature names for the transformed DataFrame if supported
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback if get_feature_names_out is not available in older scikit-learn versions
            logger.warning("get_feature_names_out not supported. Using generic feature names.")
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        # 4. Reconstruct DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        # Append the untouched target column back to the dataset
        if y is not None:
            df_transformed[target_col] = y.values
            
        logger.info("Data preprocessing completed successfully.")
        
        return df_transformed, preprocessor
        
    except Exception as e:
        logger.error(f"Error during dataset preprocessing: {e}")
        raise
