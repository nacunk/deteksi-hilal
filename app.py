import streamlit as st
import os
import cv2
import torch
import tempfile
import pandas as pd
from PIL import Image
import requests

# ==== Streamlit Config ====
st.set_page_config(page_title="Deteksi Hilal YOLOv5 + SQM/Hisab", layout="centered")

# ==== Load Model YOLOv5 (hanya sekali) ====
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", source="github")

model = load_model()
model.conf = 0.25  # threshold confidence

# ==== Buat folder output ====
os.makedirs("outputs", exist_ok=True)

# ==== Fungsi Deteksi Gambar ====
def detect_image(image_file):
    img = Image.open(image_file).convert("RGB")
    results = model(img)
    results.render()
    
    output_img_path = os.path.join("outputs", image_file.name)
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_img_path, img_bgr)
    
    # Simpan CSV & Excel
    df = results.pandas().xyxy[0]
    csv_path = os.path.splitext(output_img_path)[0] + "_detection.csv"
    excel_path = os.path.splitext(output_img_path)[0] + "_detection.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)
    
    return output_img_path, csv_path, excel_path

# ==== Fungsi Deteksi Video ====
def detect_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    input_path = tfile.name
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = os.path.join("outputs", "hilal_detected.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    all_detections = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()
        img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
        
        df = results.pandas().xyxy[0]
        df["frame"] = frame_idx
        all_detections.append(df)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    if all_detections:
        detections_df = pd.concat(all_detections, ignore_index=True)
        csv_path = os.path.join("outputs", "hilal_detected.csv")
        excel_path = os.path.join("outputs", "hilal_detected.xlsx")
        detections_df.to_csv(csv_path, index=False)
        detections_df.to_excel(excel_path, index=False)
    else:
        csv_path, excel_path = None, None
    
    return output_path, csv_path, excel_path

# ==== Fitur SQM Manual ====
sqm_value = st.number_input("Masukkan nilai SQM (mag/arcsec²):", min_value=0.0, step=0.01)
if sqm_value:
    st.write(f"Nilai SQM: {sqm_value} mag/arcsec²")

# ==== Collect Data Eksternal ====
st.subheader("Data Cuaca / Astronomi dari BMKG / API")
try:
    # Contoh: sunset/waktu maghrib Jakarta
    params = {"lat": -6.2, "lng": 106.8, "formatted": 0}
    res = requests.get("https://api.sunrise-sunset.org/json", params=params).json()
    sunset_time = res["results"]["sunset"]
    st.write(f"Waktu Sunset Jakarta (UTC): {sunset_time}")
except Exception as e:
    st.warning("Gagal mengambil data eksternal.")

# ==== Menu Deteksi ====
st.title("🌙 Deteksi Hilal Otomatis")
menu = st.radio("Pilih mode:", ["Deteksi Gambar", "Deteksi Video"])

if menu == "Deteksi Gambar":
    uploaded_image = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Gambar asli", use_container_width=True)
        with st.spinner("Mendeteksi..."):
            output_img_path, csv_path, excel_path = detect_image(uploaded_image)
        st.image(output_img_path, caption="Hasil Deteksi", use_container_width=True)
        st.success("Deteksi selesai.")
        # Download
        with open(output_img_path, "rb") as f:
            st.download_button("📷 Unduh Gambar Deteksi", f, file_name=os.path.basename(output_img_path))
        with open(csv_path, "rb") as f:
            st.download_button("📊 Unduh CSV Deteksi", f, file_name=os.path.basename(csv_path))
        with open(excel_path, "rb") as f:
            st.download_button("📑 Unduh Excel Deteksi", f, file_name=os.path.basename(excel_path))

elif menu == "Deteksi Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with st.spinner("Memproses video..."):
            output_video_path, csv_path, excel_path = detect_video(uploaded_video)
        st.video(output_video_path)
        st.success("Deteksi video selesai.")
        with open(output_video_path, "rb") as f:
            st.download_button("🎥 Unduh Video Deteksi", f, file_name="hilal_detected.mp4")
        with open(csv_path, "rb") as f:
            st.download_button("📊 Unduh CSV Deteksi", f, file_name="hilal_detected.csv")
        with open(excel_path, "rb") as f:
            st.download_button("📑 Unduh Excel Deteksi", f, file_name="hilal_detected.xlsx")
