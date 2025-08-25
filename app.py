import streamlit as st
import serial
import torch
import cv2
import os
import tempfile
import pandas as pd
from PIL import Image
import requests
from datetime import datetime

# Inisialisasi aplikasi
st.set_page_config(page_title="Deteksi Hilal + SQM + BMKG", layout="centered")
st.title("Aplikasi Deteksi Hilal dengan SQM & Cuaca BMKG")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    mdl = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", source="github")
    mdl.conf = 0.25
    return mdl

model = load_model()
os.makedirs("outputs", exist_ok=True)

# Fungsi deteksi gambar
def detect_image(image_file):
    img = Image.open(image_file).convert("RGB")
    results = model(img)
    results.render()
    fname = f"detected_{image_file.name}"
    out_img = os.path.join("outputs", fname)
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_img, img_bgr)
    df = results.pandas().xyxy[0]
    csv = out_img.rsplit('.',1)[0] + ".csv"
    xlsx = out_img.rsplit('.',1)[0] + ".xlsx"
    if not df.empty:
        df.to_csv(csv, index=False)
        df.to_excel(xlsx, index=False)
    else:
        csv, xlsx = None, None
    return out_img, csv, xlsx

# Fungsi deteksi video
def detect_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    outsv = os.path.join("outputs", "hilal_detected.mp4")
    out = cv2.VideoWriter(outsv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    all_dets, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)
        results.render()
        bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        out.write(bgr)
        df = results.pandas().xyxy[0]
        df["frame"] = frame_idx
        all_dets.append(df)
        frame_idx += 1
    cap.release(); out.release()
    if all_dets:
        df = pd.concat(all_dets, ignore_index=True)
        csv = outsv.replace(".mp4", ".csv")
        xlsx = outsv.replace(".mp4", ".xlsx")
        df.to_csv(csv, index=False)
        df.to_excel(xlsx, index=False)
    else:
        csv, xlsx = None, None
    return outsv, csv, xlsx

# SQM otomatis via USB (fallback manual)
sqm_auto = None
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    sqm_auto = ser.readline().decode().strip()
    ser.close()
except:
    sqm_auto = None

sqm_manual = st.number_input("Masukkan nilai SQM (manual, mag/arcsec²)", min_value=0.0, step=0.01)
sqm_display = sqm_auto or sqm_manual
if sqm_auto:
    st.success(f"SQM otomatis terbaca: {sqm_auto}")
elif sqm_manual:
    st.info(f"SQM manual: {sqm_manual}")

# Ambil prakiraan cuaca BMKG
st.subheader("Prakiraan Cuaca BMKG (3 hari)")
kode_kel = st.text_input("Masukkan kode wilayah (ADM IV)", placeholder="mis: 64.74.01.1006")
prakicuaca = None
if st.button("Ambil Data Cuaca"):
    if kode_kel:
        url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={kode_kel}"
        try:
            res = requests.get(url).json()
            prakicuaca = res.get("data")
            st.json(prakicuaca)
        except:
            st.error("Gagal mengambil data dari BMKG")
    else:
        st.warning("Silakan isi kode wilayah.")

# Upload deteksi
menu = st.radio("Pilih mode:", ["Deteksi Gambar", "Deteksi Video"])

if menu == "Deteksi Gambar":
    uploaded_image = st.file_uploader("Gambar Hilal", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Input", use_container_width=True)
        with st.spinner("Deteksi berlangsung..."):
            out_img, csv, xlsx = detect_image(uploaded_image)
        st.image(out_img, caption="Hasil Deteksi", use_container_width=True)
        st.download_button("Unduh Gambar", open(out_img,"rb"), file_name=os.path.basename(out_img))
        if csv and xlsx:
            st.download_button("Unduh CSV", open(csv,"rb"), file_name=os.path.basename(csv))
            st.download_button("Unduh Excel", open(xlsx,"rb"), file_name=os.path.basename(xlsx))

elif menu == "Deteksi Video":
    uploaded_video = st.file_uploader("Video Hilal", type=["mp4","avi","mov"])
    if uploaded_video:
        with st.spinner("Memproses..."):
            out_vid, csv, xlsx = detect_video(uploaded_video)
        st.video(out_vid)
        st.download_button("Unduh Video", open(out_vid,"rb"), file_name=os.path.basename(out_vid))
        if csv and xlsx:
            st.download_button("Unduh CSV", open(csv,"rb"), file_name=os.path.basename(csv))
            st.download_button("Unduh Excel", open(xlsx,"rb"), file_name=os.path.basename(xlsx))
   
