import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np

def train_model():
    """
    Executes the training pipeline: loads preprocessed data, applies feature transformation,
    trains a Deep Learning classifier, and persists the trained model artifact.
    """
    # 1. Initialize directory paths and load artifacts
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "Delivery_Logistics_Clean.csv")
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "logistics_model.h5")

    df = pd.read_csv(DATA_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # 2. Feature selection and pipeline transformation
    X = df.drop(columns=['delivery_status', 'delayed'], errors='ignore')
    
    # PERBAIKAN: Ubah nilai teks 'yes'/'no' menjadi angka 1/0
    y = df['delayed'].map({'yes': 1, 'no': 0})
    
    # Transformasi fitur
    X_processed = preprocessor.fit_transform(X).toarray()
    
    # Konversi ke float32
    X_processed = X_processed.astype('float32')
    y = y.astype('float32')

    # 3. Define Neural Network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_processed.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Model training execution
    print("Initiating model training process...")
    model.fit(X_processed, y, epochs=10, batch_size=32)

    # 5. Persist the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model successfully serialized to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()