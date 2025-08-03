import os
import cv2
import torch
import tempfile
import pandas as pd
from PIL import Image

# Pastikan YOLOv5 local clone tersedia di ./yolov5
import sys
sys.path.insert(0, os.path.abspath('./yolov5'))

from utils.datasets import letterbox
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device

# Folder output
os.makedirs("outputs", exist_ok=True)

_model = None
device = select_device('cpu')

def load_model():
    global _model
    if _model is None:
        _model = DetectMultiBackend('best.pt', device=device)
        _model.warmup()
    return _model

def detect_image(image_file):
    model = load_model()
    img = Image.open(image_file).convert('RGB')
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Resize dan preprocessing
    img_resized = letterbox(img_cv2, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = non_max_suppression(model(img_tensor)[0], conf_thres=0.25, iou_thres=0.45)
    detections = pred[0]

    df = pd.DataFrame()
    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], img_cv2.shape).round()
        df = pd.DataFrame(detections.cpu().numpy(), columns=["xmin", "ymin", "xmax", "ymax", "conf", "class"])
        df["name"] = [model.names[int(cls)] for cls in df["class"]]

        # Draw boxes
        for _, row in df.iterrows():
            cv2.rectangle(img_cv2, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (0, 255, 0), 2)
            cv2.putText(img_cv2, f'{row["name"]} {row["conf"]:.2f}', (int(row["xmin"]), int(row["ymin"])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_img_path = os.path.join("outputs", image_file.name)
    cv2.imwrite(output_img_path, img_cv2)

    csv_path = os.path.splitext(output_img_path)[0] + "_detection.csv"
    excel_path = os.path.splitext(output_img_path)[0] + "_detection.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

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

        img_resized = letterbox(frame, new_shape=640)[0]
        img_resized = img_resized.transpose((2, 0, 1))[::-1]
        img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = non_max_suppression(model(img_tensor)[0], conf_thres=0.25, iou_thres=0.45)
        detections = pred[0]

        df = pd.DataFrame()
        if detections is not None and len(detections):
            detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()
            df = pd.DataFrame(detections.cpu().numpy(), columns=["xmin", "ymin", "xmax", "ymax", "conf", "class"])
            df["frame"] = frame_idx
            df["name"] = [model.names[int(cls)] for cls in df["class"]]

            for _, row in df.iterrows():
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (0, 255, 0), 2)
                cv2.putText(frame, f'{row["name"]} {row["conf"]:.2f}', (int(row["xmin"]), int(row["ymin"])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            all_detections.append(df)

        out.write(frame)
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

        return output_path, csv_path, excel_path
    else:
        return None, None, None
