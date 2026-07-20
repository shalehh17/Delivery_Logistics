import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard(df):
    """
    Menampilkan Dashboard Analitik Logistik.
    df: Dataframe yang dikirim dari main.py (sudah dimuat via utils.load_data)
    """
    st.header("📊 Dashboard Analitik Operasional")
    
    # 1. Filter Dinamis di Sidebar
    st.sidebar.subheader("Filter Analitik")
    
    # Ambil daftar partner unik
    unique_partners = df['delivery_partner'].unique()
    partner_filter = st.sidebar.multiselect(
        "Pilih Partner Logistik:", 
        options=unique_partners, 
        default=unique_partners
    )
    
    # Validasi filter
    if not partner_filter:
        st.warning("Silakan pilih minimal satu partner logistik di sidebar.")
        return

    # Filter data berdasarkan pilihan partner
    filtered_df = df[df['delivery_partner'].isin(partner_filter)].copy()
    
    # 2. Preprocessing Kolom untuk Dashboard
    # Pastikan kolom 'delayed' ada dan diubah ke numerik untuk kalkulasi
    if 'delayed' in filtered_df.columns:
        filtered_df['is_delayed_numeric'] = filtered_df['delayed'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        filtered_df['is_delayed_numeric'] = 0

    # 3. KPI Metrics
    col1, col2, col3 = st.columns(3)
    
    delay_mean = filtered_df['is_delayed_numeric'].mean()
    col1.metric("Tingkat Keterlambatan", f"{(delay_mean*100):.1f}%")
    col2.metric("Total Partner", len(partner_filter))
    col3.metric("Total Pengiriman", len(filtered_df))
    
    st.divider()
    
    # 4. Visualisasi Tren (Plotly)
    st.subheader("Performa Partner Logistik")
    
    # Agregasi data untuk bar chart
    chart_data = filtered_df.groupby('delivery_partner')['is_delayed_numeric'].mean().reset_index()
    
    fig = px.bar(
        chart_data, 
        x='delivery_partner', 
        y='is_delayed_numeric', 
        title="Rata-rata Tingkat Keterlambatan per Partner",
        labels={'is_delayed_numeric': 'Probabilitas Delay (0-1)', 'delivery_partner': 'Partner'},
        color='is_delayed_numeric', 
        color_continuous_scale='RdYlGn_r' # Hijau (bagus) ke Merah (buruk)
    )
    
    # Mempercantik layout chart
    fig.update_layout(xaxis_title="Mitra Logistik", yaxis_title="Skala Delay")
    st.plotly_chart(fig, use_container_width=True)

    # 5. Tabel Detail (Opsional)
    if st.checkbox("Tampilkan Tabel Data"):
        st.dataframe(filtered_df[['delivery_partner', 'distance_km', 'delivery_cost', 'delayed']])