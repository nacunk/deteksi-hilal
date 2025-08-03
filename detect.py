import os
import sys
import torch
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime

# Tambahkan path ke YOLOv5 lokal
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Impor dari YOLOv5 lokal
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device


# Load model YOLOv5
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    device = select_device('cpu')
    model = DetectMultiBackend(model_path, device=device)
    return model

# Fungsi utama untuk mendeteksi hilal dari gambar
def detect_image(image_file):
    # Simpan gambar sementara
    temp_dir = Path("temp_outputs")
    temp_dir.mkdir(exist_ok=True)
    img_path = temp_dir / "input.jpg"
    with open(img_path, "wb") as f:
        f.write(image_file.getbuffer())

    # Load model
    model = load_model()
    device = model.device

    # Load gambar
    dataset = LoadImages(str(img_path), img_size=640)
    results = []
    output_img_path = temp_dir / "output.jpg"

    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference dan NMS
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45)

        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in det:
                label = f"Hilal {conf:.2f}"
                results.append({
                    "label": "hilal",
                    "confidence": float(conf),
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                    "timestamp": datetime.now().isoformat()
                })
                # Gambar bounding box
                cv2.rectangle(im0s, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(im0s, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imwrite(str(output_img_path), im0s)

    # Simpan hasil ke CSV dan Excel
    df = pd.DataFrame(results)
    csv_path = temp_dir / "results.csv"
    excel_path = temp_dir / "results.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    return str(output_img_path), str(csv_path), str(excel_path)
