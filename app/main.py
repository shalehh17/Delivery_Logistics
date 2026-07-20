import streamlit as st
import pandas as pd
import os
from utils import load_artifacts, load_data
from inference import run_inference
from dashboard import show_dashboard

# Konfigurasi Halaman
st.set_page_config(page_title="Logistics AI Concierge", layout="wide")

# 1. Model & Data Initialization
try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(f"Inference pipeline initialization failed: {e}")
    st.stop()

# 2. Application Navigation
menu = st.sidebar.radio("Navigation", ["Real-time Prediction", "Analytics Dashboard"])

if menu == "Real-time Prediction":
    st.title("🤖 Logistics AI Concierge")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Halo! Berapa jarak pengiriman (km)?"}]
        st.session_state.step = 1
        st.session_state.data = {}

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    # Logika Input Bertahap
    if st.session_state.step == 1:
        if prompt := st.chat_input("Masukkan jarak (km)..."):
            st.session_state.data['distance_km'] = float(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "Berapa berat paket (kg)?"})
            st.session_state.step = 2
            st.rerun()

    elif st.session_state.step == 2:
        if prompt := st.chat_input("Masukkan berat (kg)..."):
            st.session_state.data['package_weight_kg'] = float(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "Pilih mitra logistik di bawah:"})
            st.session_state.step = 3
            st.rerun()

    elif st.session_state.step == 3:
        partner = st.selectbox("Pilih Partner:", ["delhivery", "xpressbees", "shadowfax", "dhl", "amazon logistics", "blue dart", "fedex", "ecom express", "ekart"])
        
        if st.button("Analisis"):
            # 1. Definisikan dictionary data dengan LENGKAP 
            full_data = {
                'distance_km': float(st.session_state.data['distance_km']),
                'package_weight_kg': float(st.session_state.data['package_weight_kg']),
                'delivery_time_hours': 12,
                'expected_time_hours': 12,
                'delivery_rating': 4,
                'delivery_cost': 50000,
                'delivery_partner': partner,
                'package_type': 'Standard',
                'vehicle_type': 'Truck',
                'delivery_mode': 'Standard',
                'region': 'Jabodetabek',
                'weather_condition': 'Clear',
                'delayed': 'no',             
                'delivery_status': 'delivered'
            }
            
            # 2. Urutan kolom HARUS SAMA dengan daftar di get_config() pada preprocessing.py
            ordered_cols = [
                'distance_km', 'package_weight_kg', 'delivery_time_hours', 'expected_time_hours', 
                'delivery_rating', 'delivery_cost', 'delivery_partner', 'package_type', 
                'vehicle_type', 'delivery_mode', 'region', 'weather_condition', 
                'delayed', 'delivery_status'
            ]
            
            # 3. Buat DataFrame dengan urutan yang benar
            df_input = pd.DataFrame([full_data])[ordered_cols] 
            
            # 4. Prediksi
            is_delayed, prob, advice = run_inference(preprocessor, model, df_input)
            
            # Simpan hasil sementara ke session_state
            result_msg = f"Risiko: {prob*100:.1f}%. {advice}"
            st.session_state.messages.append({"role": "assistant", "content": result_msg})
            st.session_state.step = 4
            st.rerun()

    elif st.session_state.step == 4:
        if st.button("🔄 Mulai Sesi Baru"):
            st.session_state.messages = [{"role": "assistant", "content": "Halo! Berapa jarak pengiriman (km)?"}]
            st.session_state.step = 1
            st.session_state.data = {}
            st.rerun()

elif menu == "Analytics Dashboard":
    df = load_data() 
    show_dashboard(df)