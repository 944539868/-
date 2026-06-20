import tkinter as tk
from tkinter import ttk
import cv2
import threading
import queue
from PIL import Image, ImageTk
from ultralytics import YOLO

# ===================== 加载你的模型 =====================
MODEL_PATH = r"runs/detect/train/weights/best.pt"
MODEL_PATH = r"C:\Users\Administrator\Desktop\PythonProject\finally_model\best (1).pt"
model = YOLO(MODEL_PATH)

# 控制变量
stop_flag = False
frame_queue = queue.Queue(maxsize=3)

# ===================== 主界面 =====================
root = tk.Tk()
root.title("驾驶员行为检测 - 摄像头实时识别")
root.geometry("920x720")
root.configure(bg="#f5f5f5")

# 标题
title_label = ttk.Label(root, text="📷 摄像头实时检测", font=("微软雅黑", 16, "bold"))
title_label.pack(pady=10)

# 状
status_label = ttk.Label(root, text="状态：等待启动", font=("微软雅黑", 12))
status_label.pack(pady=5)

# 画布（显示视频）
canvas = tk.Canvas(root, width=860, height=540, bg="black")
canvas.pack(pady=10)

# ===================== 按钮 =====================
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=15)

def start_camera():
    global stop_flag
    stop_flag = False
    status_label.config(text="状态：摄像头已开启，正在检测...")
    threading.Thread(target=run_camera_detect, daemon=True).start()

def stop_camera():
    global stop_flag
    stop_flag = True
    status_label.config(text="状态：已停止")

btn_start = ttk.Button(btn_frame, text="▶ 启动摄像头", command=start_camera, width=18)
btn_start.grid(row=0, column=0, padx=15)

btn_stop = ttk.Button(btn_frame, text="■ 停止", command=stop_camera, width=18)
btn_stop.grid(row=0, column=1, padx=15)

# ===================== 摄像头检测主逻辑 =====================
def run_camera_detect():
    global stop_flag
    cap = cv2.VideoCapture(0)  # 0 = 默认摄像头
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 检测
        results = model(frame, imgsz=320)
        frame_det = results[0].plot()

        # 转 RGB
        frame_rgb = cv2.cvtColor(frame_det, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # 等比例缩放
        scale = min(860/w, 540/h)
        new_w, new_h = int(w*scale), int(h*scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # 转图像
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        # 更新画布
        canvas.delete("all")
        canvas.create_image(430, 270, image=imgtk)
        canvas.imgtk = imgtk

    cap.release()

# ===================== 运行 =====================
if __name__ == "__main__":
    root.mainloop()