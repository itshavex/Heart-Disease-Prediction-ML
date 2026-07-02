"""
Machine Learning prediction orchestrator module.
"""
import logging
import pandas as pd
from typing import Dict, Any
from app.startup import model_registry

logger = logging.getLogger(__name__)

def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a prediction from raw patient features.
    
    Args:
        features (dict): Validated input data containing patient features.
        
    Returns:
        dict: A structured dictionary containing the prediction and probabilities.
    """
    logger.info("Prediction started.")
    try:
        model = model_registry.get('model')
        preprocessor = model_registry.get('preprocessor')
        final_features = model_registry.get('final_features')
        
        if not model or not preprocessor or not final_features:
            logger.error("Model artifacts are missing from the registry.")
            raise RuntimeError("Model artifacts are not loaded in memory.")
            
        # Convert input dictionary to pandas DataFrame
        df = pd.DataFrame([features])
        
        # Apply the existing preprocessing pipeline
        X_transformed = preprocessor.transform(df)
        
        # Get feature names to reconstruct DataFrame
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Ensure only the features that survived feature engineering are passed to the model
        missing_cols = [col for col in final_features if col not in df_transformed.columns]
        if missing_cols:
            raise ValueError(f"Missing required engineered features: {missing_cols}")
            
        X_final = df_transformed[final_features]
        
        # Generate prediction
        prediction = int(model.predict(X_final)[0])
        
        # Generate probability
        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_final)[0]
            probability = float(probs[1]) # probability of positive class
            
        logger.info(f"Prediction completed successfully. Result: {prediction}")
        
        return {
            "status": "success",
            "prediction": prediction,
            "probability": probability
        }
        
    except Exception as e:
        logger.error(f"Prediction failure: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
