import streamlit as st
from detect import detect_image, detect_video
import os

st.set_page_config(page_title="Deteksi Hilal YOLOv5", layout="centered")

st.title("Aplikasi Deteksi Hilal dengan YOLOv5")

menu = st.radio("Pilih mode:", ["Deteksi Gambar", "Deteksi Video"])

if menu == "Deteksi Gambar":
    uploaded_image = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Gambar Asli", use_container_width=True)
        with st.spinner("Mendeteksi..."):
            output_img_path, csv_path, excel_path = detect_image(uploaded_image)
        st.image(output_img_path, caption="Hasil Deteksi", use_container_width=True)

        with open(output_img_path, "rb") as f:
            st.download_button("📷 Unduh Gambar Deteksi", f, file_name=os.path.basename(output_img_path))

        if csv_path:
            with open(csv_path, "rb") as f:
                st.download_button("📊 Unduh Data CSV", f, file_name=os.path.basename(csv_path))
        if excel_path:
            with open(excel_path, "rb") as f:
                st.download_button("📈 Unduh Data Excel", f, file_name=os.path.basename(excel_path))

elif menu == "Deteksi Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with st.spinner("Memproses video..."):
            output_video_path, csv_path = detect_video(uploaded_video)

        if output_video_path and os.path.exists(output_video_path):
            st.video(output_video_path)
            with open(output_video_path, "rb") as f:
                st.download_button("🎥 Unduh Video Deteksi", f, file_name=os.path.basename(output_video_path))

            if csv_path:
                with open(csv_path, "rb") as f:
                    st.download_button("📊 Unduh Data CSV", f, file_name=os.path.basename(csv_path))
        else:
            st.error("Gagal memproses video.")

