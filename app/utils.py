import os
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

def load_artifacts():
    """
    Retrieves serialized machine learning artifacts (preprocessor and model)
    using absolute path resolution for robust deployment.
    """
    # 1. Path Resolution: Determine root directory for project-wide accessibility
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 2. Define Paths: Establish target locations for serialized assets
    preprocessor_path = os.path.join(root_dir, 'models', 'preprocessor.pkl')
    model_path = os.path.join(root_dir, 'models', 'logistics_model.h5')
    
    # 3. Path Validation: Verify artifact existence before ingestion
    if not os.path.exists(preprocessor_path):
        st.error(f"DEBUG: Preprocessor artifact not found at {preprocessor_path}")
        raise FileNotFoundError(f"File {preprocessor_path} not found.")
    if not os.path.exists(model_path):
        st.error(f"DEBUG: Model artifact not found at {model_path}")
        raise FileNotFoundError(f"File {model_path} not found.")
        
    # 4. Artifact Loading: Deserialize pipeline and model objects
    preprocessor = joblib.load(preprocessor_path)
    model = tf.keras.models.load_model(model_path)
    
    return preprocessor, model

@st.cache_data
def load_data():
    """
    Ingests the refined dataset for analytical dashboarding with robust path resolution.
    """
    # 1. Path Resolution: Target the processed data directory within the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
        
    data_path = os.path.join(root_dir, 'data', 'processed', 'Delivery_Logistics_Clean.csv')
    
    # 2. Data Validation: Ensure dataset is available for EDA and dashboard consumption
    if not os.path.exists(data_path):
        st.error(f"Error: Dataset ingestion failed. Path not found at {data_path}")
        st.stop()
        
    return pd.read_csv(data_path)