import pandas as pd
import os

def clean_and_process_data(raw_filepath, processed_filepath):
    """
    Performs Extract, Transform, Load (ETL) operations on logistics data:
    cleans anomalies, handles missing values, and prepares data for downstream modeling.
    """
    print("Initiating data preprocessing pipeline...")

    # 1. Load raw dataset
    df = pd.read_csv(raw_filepath)

    # 2. Drop high-cardinality identifiers with no predictive power
    if 'delivery_id' in df.columns:
        df = df.drop(columns=['delivery_id'])

    # 3. Standardize time-series features
    # Extracting temporal hour components from formatted timestamps
    df['delivery_time_hours'] = df['delivery_time_hours'].apply(lambda x: int(str(x).split('.')[-1]))
    df['expected_time_hours'] = df['expected_time_hours'].apply(lambda x: int(str(x).split('.')[-1]))

    # 4. Data cleansing: handling nulls and removing duplicate observations
    df = df.dropna()
    df = df.drop_duplicates()

    # 5. Persist refined dataset to processed storage
    # Ensuring output directory structure exists
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df.to_csv(processed_filepath, index=False)

    print(f"Refined data successfully persisted to: {processed_filepath}")
    print(f"Post-processing dataset shape: {df.shape}")
    return df

# Main execution pipeline
if __name__ == "__main__":
    # Mendapatkan path absolut direktori tempat skrip ini berada (folder 'src')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path menuju folder data/raw dan data/processed menggunakan os.path.join
    # Skrip akan naik satu tingkat (..) dari 'src' ke root proyek, lalu masuk ke 'data'
    RAW_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "Delivery_Logistics.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "Delivery_Logistics_Clean.csv")
    
    # Cek apakah file apakah ada sebelum diproses
    if os.path.exists(RAW_PATH):
        clean_and_process_data(RAW_PATH, PROCESSED_PATH)
    else:
        print(f"Error: File tidak ditemukan di lokasi: {RAW_PATH}")
        print("Pastikan file 'Delivery_Logistics.csv' sudah dipindahkan ke folder 'data/raw/'.")

    # Blok Integrasi Download yang Aman (Portable Logic)
    try:
        from google.colab import files
        if os.path.exists(PROCESSED_PATH):
            files.download(PROCESSED_PATH)
            print(f"File '{PROCESSED_PATH}' berhasil diunduh via Colab.")
        else:
            print(f"Error: File '{PROCESSED_PATH}' tidak ditemukan.")
    except ImportError:
        # Jika bukan di Google Colab, abaikan proses unduh
        print("\n--- Diagnostic Note ---")
        print("Lingkungan lokal (VSC) terdeteksi. Modul 'google.colab' diabaikan.")
        print(f"File hasil pembersihan dapat diakses di: {os.path.abspath(PROCESSED_PATH)}")