import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Judul dan Deskripsi
st.markdown("""
    <h1 style='color:#2e7d32; text-align: center;'>🌿 Prediksi Kategori Obesitas</h1>
    <p style='text-align: center;'>Masukkan informasi gaya hidup Anda untuk melihat prediksi tingkat obesitas berdasarkan model yang telah dilatih.</p>
    <hr>
""", unsafe_allow_html=True)

# Mapping kategorikal
gender_map = {"Laki-laki": 1, "Perempuan": 0}
family_map = {"Ada": 1, "Tidak Ada": 0}
favc_map = {"Ya": 1, "Tidak": 0}
caec_map = {"Tidak Pernah": 0, "Kadang": 1, "Sering": 2, "Selalu": 3}
smoke_map = {"Ya": 1, "Tidak": 0}
scc_map = {"Ya": 1, "Tidak": 0}
calc_map = {"Tidak Pernah": 0, "Kadang": 1, "Sering": 2, "Selalu": 3}
mtrans_map = {
    "Mobil": 0,
    "Motor": 1,
    "Sepeda": 2,
    "Jalan Kaki": 3,
    "Transportasi Umum": 4
}

# Layout input
st.subheader("📋 Input Data Gaya Hidup dan Kesehatan")

col1, col2 = st.columns(2)
with col1:
    gender_label = st.selectbox("🧑 Jenis Kelamin", list(gender_map.keys()))
    age = st.slider("🎂 Umur", 10, 100, 25)
    height = st.number_input("📏 Tinggi Badan (meter)", 1.0, 2.5, 1.70)
    weight = st.number_input("⚖️ Berat Badan (kg)", 30.0, 200.0, 70.0)
    family_label = st.selectbox("👪 Riwayat Keluarga Obesitas", list(family_map.keys()))
    favc_label = st.selectbox("🍔 Konsumsi Makanan Kalori Tinggi", list(favc_map.keys()))
    fcvc = st.slider("🥗 Frekuensi Makan Sayur (1–3)", 1.0, 3.0, 2.0)
    nfc = st.slider("🍟 Fast Food (NFC)", 1.0, 3.0, 2.0)

with col2:
    caec_label = st.selectbox("🍪 Frekuensi Ngemil", list(caec_map.keys()))
    smoke_label = st.selectbox("🚬 Merokok", list(smoke_map.keys()))
    ch2o = st.slider("💧 Konsumsi Air Harian (CH2O)", 1.0, 3.0, 2.0)
    scc_label = st.selectbox("🍚 Karbohidrat Kompleks (SCC)", list(scc_map.keys()))
    faf = st.slider("🏃‍♂️ Aktivitas Fisik (FAF)", 0.0, 3.0, 1.0)
    tue = st.slider("🕒 Waktu Luang untuk Aktivitas (TUE)", 0.0, 2.0, 1.0)
    calc_label = st.selectbox("🍷 Konsumsi Alkohol (CALC)", list(calc_map.keys()))
    mtrans_label = st.selectbox("🛵 Moda Transportasi", list(mtrans_map.keys()))

# Konversi input ke numerik
gender = gender_map[gender_label]
family = family_map[family_label]
favc = favc_map[favc_label]
caec = caec_map[caec_label]
smoke = smoke_map[smoke_label]
scc = scc_map[scc_label]
calc = calc_map[calc_label]
mtrans = mtrans_map[mtrans_label]

# Susun input
input_data = np.array([[gender, age, height, weight, family, favc, fcvc, nfc, caec,
                        smoke, ch2o, scc, faf, tue, calc, mtrans]])
input_scaled = scaler.transform(input_data)

# Prediksi
st.markdown("---")
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"📊 Kategori Obesitas Anda: **{prediction}**")
    st.markdown("> ⚠️ *Hasil ini hanya sebagai referensi, bukan diagnosis medis.*")

