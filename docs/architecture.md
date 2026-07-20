# Arsitektur Pipeline MLOps & Rekayasa Fitur

## 1. Ingesti Data & Eksplorasi Analitik (EDA)
*   **Sumber:** `data/raw/Delivery_Logistics.csv` (25.000 instans, 15 atribut).
*   **Deteksi Anomali:** Mengidentifikasi dan mengekstraksi nilai integer dari parsing *datetime* yang keliru pada atribut temporal (`delivery_time_hours` dan `expected_time_hours`).

## 2. Transformasi Pra-pemrosesan (Data Preprocessing)
Pencegahan kebocoran data (*data leakage*) dilakukan melalui pemisahan data (*Train-Test Split*) awal. Transformasi dikelola secara terpusat melalui `scikit-learn Pipeline`:
*   **Fitur Numerik:** Standardisasi menggunakan `StandardScaler` (Z-score normalization) untuk mereduksi bias pada variabel berskala besar (`distance_km`, `delivery_cost`).
*   **Fitur Kategorikal:** Penerapan `OneHotEncoder` untuk mengubah atribut logistik nominal (`delivery_partner`, `weather_condition`) menjadi vektor biner orthogonal.

## 3. Pemodelan Prediktif (Modeling)
*   Implementasi algoritma klasifikasi probabilistik (seperti *Random Forest* atau *Deep Neural Networks* via TensorFlow).
*   Penyesuaian *hyperparameter* berbasis *Cross-Validation* dan ekstraksi *Feature Importance* untuk mengetahui determinan utama keterlambatan.

## 4. Serving & Visualisasi Interaktif
*   Artifak model (`.h5` atau `.joblib`) dan objek pra-pemrosesan (`.pkl`) di-*deploy* menggunakan **Streamlit**.
*   Menyediakan dua instans antarmuka: *Diagnostic Dashboard* untuk wawasan operasional dan *Inference Form* untuk input prediksi *real-time*.