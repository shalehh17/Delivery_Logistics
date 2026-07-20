import streamlit as st
import pandas as pd
import numpy as np
import traceback

def run_inference(preprocessor, model, input_data):
    """
    Menjalankan pipeline inferensi secara proaktif.
    Fungsi ini dipanggil oleh main.py setiap kali tombol analisis ditekan.
    """
    try:
        # 1. Transformasi Data
        # preprocessor.transform akan menggunakan parameter yang sudah difit saat training
        transformed_data = preprocessor.transform(input_data)
        
        # Penanganan jika output preprocessor berupa sparse matrix
        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()
        
        # Konversi ke float32 agar kompatibel dengan model TensorFlow/Keras
        transformed_data = transformed_data.astype('float32')

        # 2. Prediksi Model
        prediction = model.predict(transformed_data)
        
        # 3. Logika Penentuan Status (Delayed/Not Delayed)
        # Jika model output memiliki lebih dari 1 node (multiclass), 
        # kita ambil probabilitas kelas indeks 1 (asumsi kelas 1 = 'delayed')
        if prediction.shape[1] > 1:
            prob = float(prediction[0][1])
        else:
            prob = float(prediction[0][0])
        
        # 4. Pengambilan Keputusan Strategis
        is_delayed = prob > 0.4
        
        # Pesan saran (Advice) yang intuitif untuk manajemen operasional
        if is_delayed:
            advice = (f"⚠️ Risiko keterlambatan terdeteksi sebesar {prob*100:.0f}%. "
                      "Disarankan untuk melakukan re-alokasi mitra atau evaluasi rute alternatif.")
        else:
            advice = (f"✅ Pengiriman diprediksi aman (Risiko: {prob*100:.0f}%). "
                      "Status dalam kondisi optimal sesuai SLA.")
            
        return is_delayed, prob, advice

    except Exception as e:
        # Menangkap error jika ada ketidaksesuaian input dan mencetaknya ke terminal
        traceback.print_exc()  
        st.error(f"Error detail: {e}") 
        return False, 0.0, "Gagal menjalankan analisis prediksi."