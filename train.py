import torch
from ultralytics import YOLO
import multiprocessing
import os

os.environ["ULTRALYTICS_DOWNLOAD"] = "0"
multiprocessing.freeze_support()

if __name__ == '__main__':

    DATA_YAML = r"C:\Users\Administrator\Desktop\PythonProject\finally_model\driver_behavior.yaml"

    # 直接用官方yolov8s.pt，不依赖你本地权重
    model = YOLO("yolov8s.pt", task="detect")

    model.train(
        data=DATA_YAML,
        epochs=50,          # 足够冲到90%+
        imgsz=320,
        batch=16,
        device=0,
        workers=2,

        pretrained=True,
        amp=False,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.05,
        weight_decay=0.0005,

        # 数据增强（最稳、最提精度）
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        fliplr=0.5,

        patience=10,
        label_smoothing=0.1,
        cache="ram",

        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True
    )