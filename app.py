import streamlit as st
from detect import detect_image, detect_video
import os

# WAJIB baris pertama
st.set_page_config(page_title="Deteksi Hilal YOLOv5 + SQM/Hisab", layout="centered")

st.title("🌙 Deteksi Hilal Otomatis")
st.write("Aplikasi ini menggunakan YOLOv5 untuk mendeteksi hilal dari citra atau video observasi.")

# Input tambahan: SQM dan Hisab
with st.sidebar:
    st.header("📥 Input Tambahan")
    sqm_value = st.text_input("Masukkan nilai SQM (jika ada)", placeholder="Contoh: 20.35")
    hisab_note = st.text_area("Catatan hasil Hisab", placeholder="Contoh: Hilal diperkirakan terlihat pukul 18:15 WIB")

tab1, tab2 = st.tabs(["🖼️ Deteksi Gambar", "🎥 Deteksi Video"])

# --- Tab Deteksi Gambar ---
with tab1:
    uploaded_image = st.file_uploader("Unggah Gambar Hilal", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.spinner("Mendeteksi hilal dalam gambar..."):
            img_path, csv_path, excel_path = detect_image(uploaded_image)

            st.image(img_path, caption="Hasil Deteksi", use_container_width=True)

            st.success("Deteksi selesai.")
            st.download_button("📥 Unduh CSV", open(csv_path, "rb"), file_name=os.path.basename(csv_path))
            st.download_button("📥 Unduh Excel", open(excel_path, "rb"), file_name=os.path.basename(excel_path))

        # Tampilkan input tambahan
        if sqm_value:
            st.info(f"**Nilai SQM:** {sqm_value}")
        if hisab_note:
            st.info(f"**Catatan Hisab:** {hisab_note}")

# --- Tab Deteksi Video ---
with tab2:
    uploaded_video = st.file_uploader("Unggah Video Hilal", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        with st.spinner("Mendeteksi hilal dalam video..."):
            video_path, csv_path, excel_path = detect_video(uploaded_video)

            if video_path:
                st.video(video_path)
                st.success("Deteksi selesai pada video.")
                st.download_button("📥 Unduh CSV", open(csv_path, "rb"), file_name=os.path.basename(csv_path))
                st.download_button("📥 Unduh Excel", open(excel_path, "rb"), file_name=os.path.basename(excel_path))
            else:
                st.warning("Tidak ada deteksi hilal pada video.")

        # Tampilkan input tambahan
        if sqm_value:
            st.info(f"**Nilai SQM:** {sqm_value}")
        if hisab_note:
            st.info(f"**Catatan Hisab:** {hisab_note}")

st.markdown("---")
st.caption("🛠️ Dibuat oleh Mahasiswa Ilmu Falak | Powered by YOLOv5 + Streamlit")
