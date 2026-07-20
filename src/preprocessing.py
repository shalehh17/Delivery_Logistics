import pandas as pd
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_config():
    """Pusat definisi kolom (Single Source of Truth)"""
    # Pastikan tidak ada kolom yang sama di kedua daftar dibawah ini
    numeric_cols = ['distance_km', 'package_weight_kg', 'delivery_time_hours', 
                    'expected_time_hours', 'delivery_rating', 'delivery_cost']
    
    categorical_cols = ['delivery_partner', 'package_type', 'vehicle_type', 
                        'delivery_mode', 'region', 'weather_condition', 
                        'delayed', 'delivery_status']
    return numeric_cols, categorical_cols

def build_and_save_preprocessor(data_path, save_path):
    """Membangun, melatih, dan menyimpan preprocessor"""
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Ambil konfigurasi
    num_cols, cat_cols = get_config()
    
    # 3. PEMBERSIHAN DATA (CLEANSING)
    # Menyeragamkan format kolom agar sesuai dengan ekspektasi model
    df['delivery_cost'] = pd.to_numeric(df['delivery_cost'], errors='coerce') 
    df['delayed'] = df['delayed'].replace({'0': 'no', '1': 'yes'}) 
    df = df.dropna() 

    # Memaksa kolom kategori menjadi string/object agar tidak terbaca sebagai float
    for col in cat_cols:
        df[col] = df[col].astype(str)
        
    # Memaksa kolom numerik menjadi float
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Jika masih ada nilai kosong di kolom numerik, isi dengan 0
    df[num_cols] = df[num_cols].fillna(0)

    # 4. Definisikan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])

    # 5. Fit Preprocessor
    print("Fitting preprocessor ke dataset...")
    X = df[num_cols + cat_cols]
    preprocessor.fit(X)
    
    # 6. Simpan
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    
    print(f"Success: Preprocessor berhasil disimpan ke: {save_path}")
    return preprocessor

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_CLEAN_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "Delivery_Logistics_Clean.csv")
    SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "preprocessor.pkl")
    
    if os.path.exists(DATA_CLEAN_PATH):
        build_and_save_preprocessor(DATA_CLEAN_PATH, SAVE_PATH)
    else:
        print(f"Error: Target data file not found at {DATA_CLEAN_PATH}")