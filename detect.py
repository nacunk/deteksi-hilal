import os
import cv2
import torch
import tempfile
import pandas as pd
from PIL import Image

# =============== MODEL SETUP ===============
# Load model YOLOv5 sekali saja
_model = None

def load_model():
    global _model
    if _model is None:
        _model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", source="github")
        _model.conf = 0.25
    return _model

# Buat folder output
os.makedirs("outputs", exist_ok=True)

# =============== DETEKSI GAMBAR ===============
def detect_image(image_file, sqm_value=None, external_data: dict = None):
    """
    Deteksi hilal pada gambar.
    Tambahkan metadata SQM + external_data ke CSV/Excel jika tersedia.
    """
    model = load_model()
    img = Image.open(image_file).convert("RGB")
    results = model(img)
    results.render()

    # Simpan gambar hasil deteksi
    output_img_path = os.path.join("outputs", f"detected_{image_file.name}")
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_img_path, img_bgr)

    # Simpan hasil deteksi ke CSV/Excel
    df = results.pandas().xyxy[0]
    if not df.empty:
        # Tambahkan metadata SQM
        if sqm_value is not None:
            df["SQM"] = sqm_value
        # Tambahkan metadata eksternal (cuaca/sunset/hisab)
        if external_data:
            for k, v in external_data.items():
                df[k] = v

    csv_path = os.path.splitext(output_img_path)[0] + ".csv"
    xlsx_path = os.path.splitext(output_img_path)[0] + ".xlsx"

    if not df.empty:
        df.to_csv(csv_path, index=False)
        df.to_excel(xlsx_path, index=False)
    else:
        csv_path, xlsx_path = None, None

    return output_img_path, csv_path, xlsx_path

# =============== DETEKSI VIDEO ===============
def detect_video(video_file, sqm_value=None, external_data: dict = None):
    """
    Deteksi hilal pada video.
    Tambahkan metadata SQM + external_data ke CSV/Excel jika tersedia.
    """
    model = load_model()

    # Simpan sementara video input
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join("outputs", "hilal_detected.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
        if not df.empty:
            df["frame"] = frame_idx
            # Tambahkan metadata
            if sqm_value is not None:
                df["SQM"] = sqm_value
            if external_data:
                for k, v in external_data.items():
                    df[k] = v
            all_detections.append(df)

        frame_idx += 1

    cap.release()
    out.release()

    # Simpan CSV & Excel
    csv_path, xlsx_path = None, None
    if all_detections:
        detections_df = pd.concat(all_detections, ignore_index=True)
        csv_path = os.path.join("outputs", "hilal_detected.csv")
        xlsx_path = os.path.join("outputs", "hilal_detected.xlsx")
        detections_df.to_csv(csv_path, index=False)
        detections_df.to_excel(xlsx_path, index=False)

    return output_path, csv_path, xlsx_path
