import cv2
import numpy as np
from ultralytics import YOLO

class DriverBehaviorModel:
    def __init__(self):
        # 加载你的 YOLOv8 权重 best2.pt
        self.model = YOLO(r"weights\best2.pt")
        self.conf = 0.3
        self.imgsz = 416
        # 类别映射
        self.class_names = {
            0: "face",
            1: "smoke",
            2: "phone",
            3: "drink"
        }

    def detect(self, img):
        if img is None:
            return [], [], []
        results = self.model(
            img,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )
        res = results[0]
        labels = []
        boxes = []
        confs = []   # 新增：保存每个框的置信度

        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in self.class_names:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = int(x2 - x1)
                h = int(y2 - y1)
                labels.append(self.class_names[cls_id])
                boxes.append([int(x1), int(y1), w, h])
                confs.append(round(conf, 2))
        return labels, boxes, confs

    # 绘制框 + 类别 + 置信度
    def draw_detections(self, img, labels, boxes, confs):
        img_copy = img.copy()
        color_map = {
            "face": (255, 255, 0),
            "smoke": (0, 255, 255),
            "phone": (0, 0, 255),
            "drink": (255, 0, 255)
        }
        for label, box, conf in zip(labels, boxes, confs):
            x, y, w, h = box
            color = color_map.get(label, (0, 255, 0))
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            # 文字：类别 + 置信度
            text = f"{label} {conf}"
            cv2.putText(img_copy, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_copy