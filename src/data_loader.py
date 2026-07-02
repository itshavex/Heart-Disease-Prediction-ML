import pandas as pd
from pathlib import Path


def load_dataset() -> pd.DataFrame:
    """Load the heart disease dataset from the local storage.
    
    This function reads the raw CSV dataset into a Pandas DataFrame. It safely 
    handles potential errors such as a missing file or incorrect formatting.

    Returns:
        pd.DataFrame: The loaded dataset if successful, or None if an error occurs.
    """
    # Define the path to the dataset using pathlib for modern, clean path handling
    data_path = Path("Data") / "raw" / "heart_disease_data_1.csv"
    
    try:
        # Attempt to read the dataset
        df = pd.read_csv(data_path)
        
        # Display dataset loading success
        print("\n" + "="*60)
        print("   SUCCESS: Dataset Loaded Successfully")
        print("="*60 + "\n")
        
        return df
        
    except FileNotFoundError:
        print(f"\n[ERROR] The file was not found at: {data_path.resolve()}")
        print("Please ensure the 'heart_disease_data_1.csv' file exists in the 'Data/raw' directory.\n")
        return None
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}\n")
        return None
