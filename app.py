import streamlit as st
import os
import torch
import serial
import requests
import pandas as pd
from datetime import datetime

# -------------------
# LOAD YOLO MODEL
# -------------------
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=False)
    return model

model = load_model()

# -------------------
# DETECTION FUNCTIONS
# -------------------
def detect_image(image, model):
    results = model(image)
    return results

def detect_video(video, model):
    results = model(video)
    return results

# -------------------
# STREAMLIT APP
# -------------------
st.set_page_config(page_title="🌙 Deteksi Hilal YOLOv5 + SQM + BMKG", layout="centered")

st.title("🌙 Aplikasi Deteksi Hilal Otomatis")
st.write("Aplikasi ini mendeteksi hilal menggunakan YOLOv5, terintegrasi dengan input SQM dan data cuaca BMKG/Open-Meteo.")

# ===================
# INPUT SQM
# ===================
st.subheader("Input SQM (Sky Quality Meter)")

sqm_mode = st.radio("Pilih metode input SQM:", ["Manual", "Otomatis via USB"])
sqm_value = None

if sqm_mode == "Manual":
    sqm_value = st.number_input("Masukkan nilai SQM (mag/arcsec²)", min_value=0.0, step=0.01)
else:
    try:
        ser = serial.Serial("COM3", 9600, timeout=2)  # sesuaikan port
        raw_data = ser.readline().decode("utf-8").strip()
        sqm_value = float(raw_data) if raw_data else None
        st.success(f"SQM terbaca otomatis: {sqm_value} mag/arcsec²")
    except Exception as e:
        st.error(f"Gagal membaca SQM via USB: {e}")
        sqm_value = st.number_input("Masukkan nilai SQM secara manual (fallback)", min_value=0.0, step=0.01)

# ===================
# BMKG WEATHER DATA
# ===================
st.subheader("Data Cuaca dari BMKG/Open-Meteo")

cities = {
    "Jakarta": (-6.2, 106.8),
    "Bandung": (-6.9, 107.6),
    "Surabaya": (-7.2, 112.7),
}
selected_city = st.selectbox("Pilih lokasi:", list(cities.keys()))
lat, lon = cities[selected_city]

temp, wind = None, None
try:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    r = requests.get(url)
    data = r.json()
    temp = data["current_weather"]["temperature"]
    wind = data["current_weather"]["windspeed"]
    st.write(f"📍 Lokasi: {selected_city}")
    st.write(f"🌡 Suhu: {temp} °C")
    st.write(f"💨 Kecepatan Angin: {wind} km/h")
except Exception as e:
    st.error(f"Gagal mengambil data cuaca: {e}")

# ===================
# YOLO DETECTION
# ===================
st.subheader("Deteksi Hilal")

uploaded_file = st.file_uploader("Upload gambar atau video untuk deteksi", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

detection_results = None
hilal_count = 0

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext in ["jpg", "jpeg", "png"]:
        st.image(uploaded_file, caption="Input Image", use_column_width=True)
        detection_results = detect_image(uploaded_file, model)
        st.image(detection_results.render()[0], caption="Hasil Deteksi", use_column_width=True)
        hilal_count = int((detection_results.pred[0][:, -1] == 0).sum().item())  # asumsikan class=0 adalah hilal
    else:
        st.video(uploaded_file)
        detection_results = detect_video(uploaded_file, model)
        st.write("Deteksi pada video selesai.")
        hilal_count = int((detection_results.pred[0][:, -1] == 0).sum().item())

    st.success(f"Jumlah hilal terdeteksi: {hilal_count}")

    # ===================
    # SAVE RESULTS TO CSV
    # ===================
    st.subheader("Simpan Hasil Deteksi")

    result_data = {
        "waktu_deteksi": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "SQM": [sqm_value],
        "lokasi": [selected_city],
        "suhu": [temp],
        "angin": [wind],
        "jumlah_hilal": [hilal_count],
    }

    df = pd.DataFrame(result_data)

    csv_file = "hasil_deteksi.csv"

    if os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(csv_file, index=False)

    st.download_button(
        label="⬇️ Download Hasil Deteksi (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_deteksi.csv",
        mime="text/csv",
    )
