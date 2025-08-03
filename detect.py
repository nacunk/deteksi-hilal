import os
import cv2
import torch
import tempfile
import pandas as pd
from PIL import Image

# Folder output
os.makedirs("outputs", exist_ok=True)

# Load YOLOv5 model hanya sekali
_model = None
def load_model():
    global _model
    if _model is None:
        _model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", source="github")
        _model.conf = 0.25
    return _model

def detect_image(image_file):
    model = load_model()
    img = Image.open(image_file)
    results = model(img)
    results.render()

    output_img_path = os.path.join("outputs", image_file.name)
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_img_path, img_bgr)

    df = results.pandas().xyxy[0]
    csv_path = os.path.splitext(output_img_path)[0] + "_detection.csv"
    excel_path = os.path.splitext(output_img_path)[0] + "_detection.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    # Optional: Upload ke Google Drive / Firestore (komentar dulu)
    # upload_to_cloud(csv_path)

    return output_img_path, csv_path, excel_path

def detect_video(video_file):
    model = load_model()
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

    if os.path.exists(output_path):
        csv_path = os.path.join("outputs", "hilal_video_detection.csv")
        excel_path = os.path.join("outputs", "hilal_video_detection.xlsx")
        if all_detections:
            detections_df = pd.concat(all_detections, ignore_index=True)
            detections_df.to_csv(csv_path, index=False)
            detections_df.to_excel(excel_path, index=False)
        else:
            csv_path, excel_path = None, None

        # Optional: Upload ke cloud
        # upload_to_cloud(csv_path)

        return output_path, csv_path, excel_path
    else:
        return None, None, None

# --- Opsional Upload ke Cloud (aktifkan bila siap) ---
# def upload_to_cloud(file_path):
#     from google.cloud import storage
#     client = storage.Client.from_service_account_json("path_to_your_credentials.json")
#     bucket = client.bucket("nama_bucket_anda")
#     blob = bucket.blob(os.path.basename(file_path))
#     blob.upload_from_filename(file_path)
#     print(f"Uploaded {file_path} to Google Cloud Storage")
