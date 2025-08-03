import os
import torch
import cv2
import pandas as pd
from PIL import Image
import tempfile

os.makedirs("outputs", exist_ok=True)

_model = None

def load_model():
    global _model
    if _model is None:
        _model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')
        _model.conf = 0.25
    return _model

def detect_image(image_file):
    model = load_model()
    img = Image.open(image_file).convert('RGB')

    results = model(img)
    results.render()

    output_img_path = os.path.join("outputs", f"detected_{image_file.name}")
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_img_path, img_bgr)

    df = results.pandas().xyxy[0]
    csv_path = output_img_path.rsplit('.',1)[0] + ".csv"
    excel_path = output_img_path.rsplit('.',1)[0] + ".xlsx"
    if not df.empty:
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
    else:
        csv_path = None
        excel_path = None

    return output_img_path, csv_path, excel_path

def detect_video(video_file):
    model = load_model()

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join("outputs", "detected_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
        csv_path = output_path.rsplit('.',1)[0] + ".csv"
        detections_df.to_csv(csv_path, index=False)
    else:
        csv_path = None

    return output_path, csv_path
