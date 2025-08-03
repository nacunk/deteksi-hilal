import os
import cv2
import torch
import tempfile
import pandas as pd
from PIL import Image
import sys

# Tambahkan path ke YOLOv5
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device
from utils.plots import plot_one_box

# Buat folder output
os.makedirs("outputs", exist_ok=True)

# Load model hanya sekali
_model = None
device = select_device('')  # Gunakan CPU atau GPU jika tersedia

def load_model():
    global _model
    if _model is None:
        weights = os.path.join(YOLOV5_PATH, "best.pt")
        _model = attempt_load(weights, map_location=device)
        _model.eval()
    return _model

def detect_image(image_file):
    model = load_model()
    img = Image.open(image_file).convert('RGB')
    img_path = os.path.join("outputs", image_file.name)
    img.save(img_path)

    dataset = LoadImages(img_path, img_size=640)
    names = model.module.names if hasattr(model, 'module') else model.names
    detections = []

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in pred:
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=(255, 0, 0), line_thickness=2)
                detections.append({
                    "xmin": int(xyxy[0]),
                    "ymin": int(xyxy[1]),
                    "xmax": int(xyxy[2]),
                    "ymax": int(xyxy[3]),
                    "confidence": float(conf),
                    "class": int(cls),
                    "label": names[int(cls)]
                })

        out_img_path = os.path.join("outputs", f"detected_{os.path.basename(path)}")
        cv2.imwrite(out_img_path, im0s)

        df = pd.DataFrame(detections)
        csv_path = out_img_path.replace(".jpg", ".csv").replace(".png", ".csv")
        excel_path = csv_path.replace(".csv", ".xlsx")
        if not df.empty:
            df.to_csv(csv_path, index=False)
            df.to_excel(excel_path, index=False)
        else:
            csv_path, excel_path = None, None

        return out_img_path, csv_path, excel_path

    return None, None, None
