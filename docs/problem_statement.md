# Business Problem Statement & Analytic Scope

## 1. Latar Belakang & Objektif
Optimalisasi rantai pasok membutuhkan deteksi dini terhadap potensi kegagalan SLA (*Service Level Agreement*). Objektif dari proyek ini adalah membangun model *Machine Learning* prediktif untuk mengklasifikasikan probabilitas keterlambatan (*delay*) pada pengiriman paket.

## 2. Identifikasi Variabel Utama (Features & Target)
*   **Variabel Target (Y):** `delivery_status` (Klasifikasi Multikelas: *delivered*, *delayed*, *failed*).
*   **Fitur Prediktor (X):** 
    *   **Numerik Kontinu:** `distance_km`, `package_weight_kg`, `delivery_cost`.
    *   **Kategorikal Nominal:** `delivery_partner`, `vehicle_type`, `weather_condition`, `region`.
    *   **Kategorikal Ordinal:** `delivery_rating`.

## 3. Metrik Evaluasi (KPI)
Mengingat adanya potensi ketidakseimbangan kelas (*class imbalance*) antara paket yang berhasil dikirim dan yang gagal/terlambat, akurasi matriks tunggal tidak cukup. Model akan dievaluasi menggunakan metrik pembobotan:
*   **Macro F1-Score:** Memastikan performa presisi dan sensitivitas (Recall) yang seimbang pada kelas minoritas (*failed/delayed*).
*   **ROC-AUC (One-vs-Rest):** Mengukur kemampuan model dalam memisahkan probabilitas antar kelas.