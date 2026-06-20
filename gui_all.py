# gui.py
import tkinter as tk
from tkinter import filedialog
import cv2
import time
import numpy as np
from PIL import Image, ImageTk
from model1 import DriverBehaviorModel
from model2 import FatigueDetector

class IntegratedMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("驾驶员状态综合检测系统")
        self.root.geometry("1280x780")
        self.root.configure(bg="#f0f2f5")

        self.model1 = DriverBehaviorModel()
        self.model2 = FatigueDetector()

        self.cap = None
        self.current_img = None
        self.media_type = None
        self.is_detecting = False

        self.YAWN_SCALE = 1.6
        self.YAWN_DURATION = 1.0

        self.blink_count = 0
        self.blink_times = []
        self.eye_close_streak = 0
        self.max_close_duration = 0
        self.current_close_start = 0
        self.detect_start_time = 0
        self.yawn_count = 0
        self.in_yawn = False
        self.yawn_counted = False
        self.yawn_start_time = 0

        self.BLINK_FREQ_THRESH = 25
        self.YAWN_LIGHT = 2
        self.LONG_CLOSE_THRESH = 1.0
        self.TIME_WINDOW = 60.0

        self.close_record = []

        self.frame_cnt = 0
        self.last_fps_print = time.time()
        self.fps_print_interval = 5.0

        self.calib_frame_target = 50
        self.calib_ear_list = []
        self.is_calibrated = False
        self.base_ear = 0.0
        self.ear_normal_thresh = 0.0
        self.ear_half_thresh = 0.0

        self.pitch_threshold = -2.0
        self.yaw_threshold = 20.0            # 放宽偏航限制
        self.face_angle_threshold = 15.0     # 新增倾斜角限制
        self.mar_normal_range = self.model2.MAR_NORMAL
        self.mar_side_range = self.model2.MAR_SIDE

        self.use_default_thresh = False

        self.eye_left_idx  = list(range(36, 42))
        self.eye_right_idx = list(range(42, 48))
        self.mouth_idx     = [48,50,52,54,56,58]
        self.red_point_indices = set()
        for idx in self.eye_left_idx:
            self.red_point_indices.add(idx)
        for idx in self.eye_right_idx:
            self.red_point_indices.add(idx)
        for idx in self.mouth_idx:
            self.red_point_indices.add(idx)

        self.last_angle_update_time = 0
        self.angle_update_interval = 0.2
        self.angle_smooth_factor = 0.6

        self.smooth_face_angle = 0.0
        self.smooth_pitch_angle = 0.0
        self.smooth_yaw_angle = 0.0

        self.smooth_ex1 = 0.0
        self.smooth_ey1 = 0.0
        self.bbox_smooth_factor = 0.3
        self.bbox_initialized = False

        self.last_mouth_update_time = 0
        self.mouth_update_interval = 0.4
        self.cached_mar = 0.0
        self.cached_mouth_state = "未检测"

        self.eye_close_start_time = 0
        self.long_eye_close = False

        self.setup_ui()

    def setup_ui(self):
        self.left_main = tk.Frame(self.root, bg="white", bd=2, relief=tk.RIDGE)
        self.left_main.place(x=20, y=20, width=800, height=620)
        tk.Label(self.left_main, text="检测画面", font=("微软雅黑", 14, "bold"), bg="white").pack(pady=5)
        self.canvas = tk.Canvas(self.left_main, bg="#e6e6e6", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        status_frame = tk.Frame(self.root, bg="white", bd=2, relief=tk.RIDGE)
        status_frame.place(x=20, y=650, width=800, height=100)
        self.level_lab = tk.Label(status_frame, text="驾驶员清醒", font=("微软雅黑", 20, "bold"), fg="black", bg="white")
        self.level_lab.pack(fill=tk.X, pady=8)
        self.state_lab = tk.Label(status_frame, text="行为正常", font=("微软雅黑", 16, "bold"), fg="black", bg="white")
        self.state_lab.pack(fill=tk.X, pady=4)

        right_panel = tk.Frame(self.root, bg="white", bd=2, relief=tk.RIDGE)
        right_panel.place(x=840, y=20, width=400, height=740)

        btn_w = 110
        btn_h = 35
        self.btn_photo = tk.Button(right_panel, text="选择照片", font=("微软雅黑", 10),
                                   width=12, height=2, command=self.load_photo)
        self.btn_photo.place(x=30, y=30, width=btn_w, height=btn_h)
        self.btn_video = tk.Button(right_panel, text="选择视频", font=("微软雅黑", 10),
                                   width=12, height=2, command=self.load_video)
        self.btn_video.place(x=160, y=30, width=btn_w, height=btn_h)
        self.btn_camera = tk.Button(right_panel, text="打开摄像头", font=("微软雅黑", 10),
                                    width=12, height=btn_h, command=self.open_camera)
        self.btn_camera.place(x=30, y=80, width=240, height=btn_h)
        self.btn_start = tk.Button(right_panel, text="开始检测", bg="#27ae60", fg="white",
                                  font=("微软雅黑", 10), width=12, height=2, command=self.start_detect)
        self.btn_start.place(x=30, y=130, width=btn_w, height=btn_h)
        self.btn_stop = tk.Button(right_panel, text="停止检测", bg="#e74c3c", fg="white",
                                 font=("微软雅黑", 10), width=12, height=2, command=self.stop_detect)
        self.btn_stop.place(x=160, y=130, width=btn_w, height=btn_h)
        self.btn_reset = tk.Button(right_panel, text="重置统计", bg="#3498db", fg="white",
                                   font=("微软雅黑", 10), width=12, height=2, command=self.reset_all_data)
        self.btn_reset.place(x=30, y=180, width=240, height=btn_h)

        param_frame = tk.Frame(right_panel, bg="white")
        param_frame.place(x=20, y=230, width=360, height=510)
        tk.Label(param_frame, text="疲劳检测", font=("微软雅黑", 12, "bold"), bg="white", fg="#2c3e50")\
            .pack(anchor="w", pady=5)

        self.ear_collect_lab = tk.Label(param_frame, text="初始EAR采集：未生效", font=("微软雅黑", 10), bg="white")
        self.ear_collect_lab.pack(anchor="w", pady=2)

        row2 = tk.Frame(param_frame, bg="white")
        row2.pack(anchor="w", pady=2)
        tk.Label(row2, text="眼睛状态：", font=("微软雅黑", 10), bg="white").grid(row=0, column=0)
        self.eye_state_lab = tk.Label(row2, text="", font=("微软雅黑", 10), bg="white")
        self.eye_state_lab.grid(row=0, column=1, padx=5)
        tk.Label(row2, text="嘴巴动作：", font=("微软雅黑", 10), bg="white").grid(row=0, column=2, padx=(10,0))
        self.mouth_state_lab = tk.Label(row2, text="", font=("微软雅黑", 10), bg="white")
        self.mouth_state_lab.grid(row=0, column=3, padx=5)

        row3 = tk.Frame(param_frame, bg="white")
        row3.pack(anchor="w", pady=2)
        tk.Label(row3, text="脸部角度：", font=("微软雅黑", 10), bg="white").grid(row=0, column=0)
        self.face_angle_lab = tk.Label(row3, text="0.0°", font=("微软雅黑", 10), bg="white")
        self.face_angle_lab.grid(row=0, column=1, padx=5)
        tk.Label(row3, text="头部仰角：", font=("微软雅黑", 10), bg="white").grid(row=0, column=2, padx=(10,0))
        self.pitch_angle_lab = tk.Label(row3, text="0.0°", font=("微软雅黑", 10), bg="white")
        self.pitch_angle_lab.grid(row=0, column=3, padx=5)

        self.blink_cnt_lab = tk.Label(param_frame, text="眨眼次数：0", font=("微软雅黑", 10), bg="white")
        self.blink_cnt_lab.pack(anchor="w", pady=2)
        self.yawn_cnt_lab = tk.Label(param_frame, text="哈欠次数：0", font=("微软雅黑", 10), bg="white")
        self.yawn_cnt_lab.pack(anchor="w", pady=2)
        self.blink_freq_lab = tk.Label(param_frame, text="眨眼频率(次/分钟)：0", font=("微软雅黑", 10), bg="white")
        self.blink_freq_lab.pack(anchor="w", pady=2)
        self.max_eye_close_lab = tk.Label(param_frame, text="一分钟内最长闭眼：0s", font=("微软雅黑", 10), bg="white")
        self.max_eye_close_lab.pack(anchor="w", pady=2)

        tk.Label(param_frame, text="分心检测", font=("微软雅黑", 12, "bold"), bg="white", fg="#2c3e50")\
            .pack(anchor="w", pady=(15,5))
        self.behavior_lab = tk.Label(param_frame, text="行为：无", font=("微软雅黑", 10), bg="white", fg="black")
        self.behavior_lab.pack(anchor="w", pady=2)

    def set_select_btn_state(self, state):
        self.btn_photo.config(state=state)
        self.btn_video.config(state=state)
        self.btn_camera.config(state=state)

    def load_photo(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg;*.png;*.jpeg")])
        if not path:
            return
        self.reset_all_data()
        self.cap = None
        self.media_type = "photo"
        self.use_default_thresh = True
        self.current_img = cv2.imread(path)
        self.ear_collect_lab.config(text="使用默认EAR阈值")
        self.start_detect()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("视频", "*.mp4;*.avi;*.mov;*.mkv")])
        if not path:
            return
        self.reset_all_data()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.media_type = "video"
        self.use_default_thresh = False
        self.current_img = None
        self.start_detect()

    def open_camera(self):
        self.reset_all_data()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.media_type = "camera"
        self.use_default_thresh = False
        self.current_img = None
        self.start_detect()

    def start_detect(self):
        if self.is_detecting or self.media_type is None:
            return
        self.is_detecting = True
        self.set_select_btn_state(tk.DISABLED)
        self.detect_start_time = time.time()
        self.frame_cnt = 0
        self.last_fps_print = time.time()
        self.last_angle_update_time = 0
        self.last_mouth_update_time = 0
        self.bbox_initialized = False
        self.eye_close_start_time = 0
        self.long_eye_close = False
        if self.media_type == "photo":
            self.process_photo()
        else:
            self.update_stream_frame()

    def stop_detect(self):
        self.is_detecting = False
        self.set_select_btn_state(tk.NORMAL)

    def reset_all_data(self):
        self.blink_count = 0
        self.blink_times.clear()
        self.eye_close_streak = 0
        self.max_close_duration = 0
        self.current_close_start = 0
        self.yawn_count = 0
        self.in_yawn = False
        self.yawn_counted = False
        self.yawn_start_time = 0
        self.close_record.clear()

        self.calib_ear_list.clear()
        self.is_calibrated = False
        self.base_ear = 0.0
        self.ear_normal_thresh = 0.0
        self.ear_half_thresh = 0.0

        self.eye_state_lab.config(text="")
        self.mouth_state_lab.config(text="未检测")
        self.face_angle_lab.config(text="0.0°")
        self.pitch_angle_lab.config(text="0.0°")
        self.blink_cnt_lab.config(text="眨眼次数：0")
        self.yawn_cnt_lab.config(text="哈欠次数：0")
        self.blink_freq_lab.config(text="眨眼频率(次/分钟)：0")
        self.max_eye_close_lab.config(text="一分钟内最长闭眼：0s")
        self.behavior_lab.config(text="行为：无", fg="black")
        self.level_lab.config(text="驾驶员清醒", fg="black")
        self.state_lab.config(text="行为正常", fg="black")

        self.smooth_face_angle = 0.0
        self.smooth_pitch_angle = 0.0
        self.smooth_yaw_angle = 0.0
        self.last_angle_update_time = 0
        self.last_mouth_update_time = 0
        self.cached_mar = 0.0
        self.cached_mouth_state = "未检测"
        self.bbox_initialized = False
        self.eye_close_start_time = 0
        self.long_eye_close = False

    def expand_bbox(self, x1, y1, x2, y2, img_w, img_h, scale=1.6):
        w = x2 - x1
        h = y2 - y1
        dw = int(w * (scale - 1) / 2)
        dh = int(h * (scale - 1) / 2)
        nx1 = max(0, x1 - dw)
        ny1 = max(0, y1 - dh)
        nx2 = min(img_w, x2 + dw)
        ny2 = min(img_h, y2 + dh)
        return nx1, ny1, nx2, ny2

    def draw_landmarks(self, img, landmarks, offset_x, offset_y):
        for idx, (x, y) in enumerate(landmarks):
            px = int(x + offset_x)
            py = int(y + offset_y)
            if idx in self.red_point_indices:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.circle(img, (px, py), 2, color, -1)

    def calibrate_ear(self, ear_avg, pitch, yaw, face_angle, mar):
        if self.use_default_thresh or self.is_calibrated:
            return
        if (pitch is not None and pitch > self.pitch_threshold and
            yaw is not None and abs(yaw) < self.yaw_threshold and
            face_angle is not None and face_angle < self.face_angle_threshold and
            mar < self.mar_normal_range):
            self.calib_ear_list.append(ear_avg)
            current_num = len(self.calib_ear_list)
            self.ear_collect_lab.config(text=f"初始EAR采集中：{current_num}/{self.calib_frame_target}")
            if current_num >= self.calib_frame_target:
                self.base_ear = np.mean(self.calib_ear_list)
                self.ear_normal_thresh = self.base_ear * 0.8
                self.ear_half_thresh = self.base_ear * 0.5
                self.is_calibrated = True
                self.ear_collect_lab.config(text=f"初始EAR采集生效 | 基准值：{self.base_ear:.3f}")

    def get_class_color(self, cls_name):
        if cls_name == "face":
            return (0, 255, 0)
        elif cls_name == "phone":
            return (0, 0, 255)
        elif cls_name == "smoke":
            return (0, 140, 255)
        elif cls_name == "drink":
            return (255, 0, 0)
        else:
            return (255, 255, 0)

    def draw_label_big(self, img, cls_list, box_list, conf_list):
        img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 4
        for cls, box, conf in zip(cls_list, box_list, conf_list):
            color = self.get_class_color(cls)
            x, y, bw, bh = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255,255,255), -1)
            cv2.putText(img, text, (x1 + 5, y1 - 5), font, font_scale, color, thickness)
        return img

    def draw_label_normal(self, img, cls_list, box_list, conf_list):
        img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        for cls, box, conf in zip(cls_list, box_list, conf_list):
            color = self.get_class_color(cls)
            x, y, bw, bh = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{cls} {conf:.2f}"
            cv2.putText(img, text, (x1, y1 - 5), font, font_scale, color, thickness)
        return img

    def process_photo(self):
        img = self.current_img.copy()
        det_ret = self.model1.detect(img)
        cls_list, box_list, conf_list = det_ret[0], det_ret[1], det_ret[2]
        img_draw = self.draw_label_big(img, cls_list, box_list, conf_list)

        eye_state = "未检测"
        mouth_state = "未检测"
        face_angle = 0.0
        pitch_angle = 0.0
        yaw_angle = 0.0
        landmarks = None
        h_img, w_img = img.shape[:2]

        behaviors = []
        if "phone" in cls_list:
            behaviors.append("玩手机")
        if "smoke" in cls_list:
            behaviors.append("抽烟")
        if "drink" in cls_list:
            behaviors.append("喝水")
        if pitch_angle < -4:
            behaviors.append("低头分心")

        if behaviors:
            text = "行为：" + " | ".join(behaviors)
            self.behavior_lab.config(text=text, fg="red", font=("微软雅黑", 11, "bold"))
        else:
            self.behavior_lab.config(text="行为：无", fg="black", font=("微软雅黑", 10))

        face_box = None
        for cls, box in zip(cls_list, box_list):
            if cls == "face":
                x, y, w, h = box
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                face_box = (x1, y1, x2, y2)
                break

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            ex1, ey1, ex2, ey2 = self.expand_bbox(x1, y1, x2, y2, w_img, h_img, self.YAWN_SCALE)
            if ex2 > ex1 and ey2 > ey1:
                face_crop = img[ey1:ey2, ex1:ex2]
                h_crop, w_crop = face_crop.shape[:2]
                if w_crop >= 60 and h_crop >= 60:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    landmarks = self.model2.get_landmarks(face_crop_rgb)
                    if landmarks is not None:
                        self.draw_landmarks(img_draw, landmarks, ex1, ey1)
                        landmarks_global = landmarks.copy()
                        landmarks_global[:, 0] += ex1
                        landmarks_global[:, 1] += ey1
                        face_angle = self.model2.get_face_angle(landmarks)
                        p_ret = self.model2.get_pitch_angle(landmarks_global, img.shape)
                        pitch_angle = p_ret if p_ret is not None else 0.0
                        y_ret = self.model2.get_yaw_angle(landmarks_global, img.shape)
                        yaw_angle = y_ret if y_ret is not None else 0.0

        if landmarks is not None:
            ear_avg = 0.0
            mar = 0.0
            try:
                ear_l = self.model2.calculate_ear(landmarks[self.eye_left_idx])
                ear_r = self.model2.calculate_ear(landmarks[self.eye_right_idx])
                ear_avg = (ear_l + ear_r) / 2
                mar = self.model2.calculate_mar(landmarks[self.mouth_idx])
            except:
                pass
            self.calibrate_ear(ear_avg, pitch_angle, yaw_angle, face_angle, mar)
            if ear_avg > self.model2.EAR_NORMAL:
                eye_state = "眼睛睁开"
            elif ear_avg > self.model2.EAR_HALF:
                eye_state = "眼睛半闭"
            else:
                eye_state = "眼睛闭合"
            if mar > self.mar_normal_range:
                mouth_state = "嘴巴张开"
            else:
                mouth_state = "嘴巴闭合"

        self.face_angle_lab.config(text=f"{face_angle:.1f}°")
        self.pitch_angle_lab.config(text=f"{pitch_angle:.1f}°")

        self.eye_state_lab.config(text=eye_state)
        self.mouth_state_lab.config(text=mouth_state)

        now = time.time()
        run_dur = now - self.detect_start_time
        blink_freq = (self.blink_count/run_dur)*60 if run_dur>0 else 0
        self.update_status_line(eye_state, mouth_state, now, blink_freq, behaviors)
        self.show_frame(img_draw)
        self.is_detecting = False
        self.set_select_btn_state(tk.NORMAL)

    def update_stream_frame(self):
        if not self.is_detecting or not self.cap:
            return
        ret, img = self.cap.read()
        if not ret:
            self.stop_detect()
            return
        now = time.time()
        self.frame_cnt += 1

        if now - self.last_fps_print >= self.fps_print_interval:
            fps = self.frame_cnt / (now - self.last_fps_print)
            print(f"当前帧率: {fps:.1f} FPS")
            self.frame_cnt = 0
            self.last_fps_print = now

        det_ret = self.model1.detect(img)
        cls_list, box_list, conf_list = det_ret[0], det_ret[1], det_ret[2]
        img_draw = self.draw_label_normal(img, cls_list, box_list, conf_list)

        eye_state = "未检测"
        mouth_state = self.cached_mouth_state
        face_angle = self.smooth_face_angle
        pitch_angle = self.smooth_pitch_angle
        yaw_angle = self.smooth_yaw_angle
        landmarks = None
        h_img, w_img = img.shape[:2]

        behaviors = []
        if "phone" in cls_list:
            behaviors.append("玩手机")
        if "smoke" in cls_list:
            behaviors.append("抽烟")
        if "drink" in cls_list:
            behaviors.append("喝水")
        if self.smooth_pitch_angle < -4:
            behaviors.append("低头分心")

        if behaviors:
            text = "行为：" + " | ".join(behaviors)
            self.behavior_lab.config(text=text, fg="red", font=("微软雅黑", 11, "bold"))
        else:
            self.behavior_lab.config(text="行为：无", fg="black", font=("微软雅黑", 10))

        face_box = None
        for cls, box in zip(cls_list, box_list):
            if cls == "face":
                x, y, w, h = box
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                face_box = (x1, y1, x2, y2)
                break

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            ex1, ey1, ex2, ey2 = self.expand_bbox(x1, y1, x2, y2, w_img, h_img, self.YAWN_SCALE)

            if not self.bbox_initialized:
                self.smooth_ex1 = float(ex1)
                self.smooth_ey1 = float(ey1)
                self.bbox_initialized = True
            else:
                self.smooth_ex1 = self.bbox_smooth_factor * ex1 + (1 - self.bbox_smooth_factor) * self.smooth_ex1
                self.smooth_ey1 = self.bbox_smooth_factor * ey1 + (1 - self.bbox_smooth_factor) * self.smooth_ey1

            smooth_x1 = int(round(self.smooth_ex1))
            smooth_y1 = int(round(self.smooth_ey1))
            smooth_x2 = smooth_x1 + (ex2 - ex1)
            smooth_y2 = smooth_y1 + (ey2 - ey1)

            smooth_x1 = max(0, smooth_x1)
            smooth_y1 = max(0, smooth_y1)
            smooth_x2 = min(w_img, smooth_x2)
            smooth_y2 = min(h_img, smooth_y2)

            if smooth_x2 > smooth_x1 and smooth_y2 > smooth_y1:
                face_crop = img[smooth_y1:smooth_y2, smooth_x1:smooth_x2]
                h_crop, w_crop = face_crop.shape[:2]
                if w_crop >= 60 and h_crop >= 60:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    landmarks = self.model2.get_landmarks(face_crop_rgb)
                    if landmarks is not None:
                        self.draw_landmarks(img_draw, landmarks, smooth_x1, smooth_y1)
                        if now - self.last_angle_update_time >= self.angle_update_interval:
                            landmarks_global = landmarks.copy()
                            landmarks_global[:, 0] += smooth_x1
                            landmarks_global[:, 1] += smooth_y1

                            raw_face = self.model2.get_face_angle(landmarks)
                            raw_pitch = self.model2.get_pitch_angle(landmarks_global, img.shape)
                            raw_yaw = self.model2.get_yaw_angle(landmarks_global, img.shape)

                            if raw_face is not None:
                                self.smooth_face_angle = self.angle_smooth_factor * raw_face + (1 - self.angle_smooth_factor) * self.smooth_face_angle
                            if raw_pitch is not None:
                                self.smooth_pitch_angle = self.angle_smooth_factor * raw_pitch + (1 - self.angle_smooth_factor) * self.smooth_pitch_angle
                            if raw_yaw is not None:
                                self.smooth_yaw_angle = self.angle_smooth_factor * raw_yaw + (1 - self.angle_smooth_factor) * self.smooth_yaw_angle

                            face_angle = self.smooth_face_angle
                            pitch_angle = self.smooth_pitch_angle
                            yaw_angle = self.smooth_yaw_angle

                            self.last_angle_update_time = now
                            self.face_angle_lab.config(text=f"{face_angle:.1f}°")
                            self.pitch_angle_lab.config(text=f"{pitch_angle:.1f}°")

        if landmarks is not None:
            ear_avg = 0.0
            try:
                ear_l = self.model2.calculate_ear(landmarks[self.eye_left_idx])
                ear_r = self.model2.calculate_ear(landmarks[self.eye_right_idx])
                ear_avg = (ear_l + ear_r) / 2
            except:
                pass

            self.calibrate_ear(ear_avg, pitch_angle, yaw_angle, face_angle, self.cached_mar)

            if self.is_calibrated:
                if ear_avg > self.ear_normal_thresh:
                    if self.eye_close_streak >= 2:
                        self.blink_count += 1
                        self.blink_times.append(now)
                    if self.current_close_start > 0:
                        close_dur = now - self.current_close_start
                        self.close_record.append((now, close_dur))
                        self.current_close_start = 0
                    self.eye_close_streak = 0
                    self.eye_close_start_time = 0
                    self.long_eye_close = False
                    eye_state = "眼睛睁开"
                else:
                    if ear_avg > self.ear_half_thresh:
                        eye_state = "眼睛半闭"
                    else:
                        eye_state = "眼睛闭合"
                    self.eye_close_streak += 1
                    if self.current_close_start == 0:
                        self.current_close_start = now

                    if eye_state == "眼睛闭合":
                        if self.eye_close_start_time == 0:
                            self.eye_close_start_time = now
                        if now - self.eye_close_start_time >= 0.3:
                            self.long_eye_close = True
                    else:
                        if self.eye_close_start_time == 0:
                            self.eye_close_start_time = now
            else:
                eye_state = "采集中，暂不判定"
                self.eye_close_start_time = 0
                self.long_eye_close = False

            if now - self.last_mouth_update_time >= self.mouth_update_interval:
                mar = 0.0
                try:
                    mar = self.model2.calculate_mar(landmarks[self.mouth_idx])
                except:
                    pass
                self.cached_mar = mar

                if mar > self.mar_normal_range:
                    mouth_state = "嘴巴张开"
                    if not self.in_yawn:
                        self.in_yawn = True
                        self.yawn_start_time = now
                    if self.in_yawn and not self.yawn_counted:
                        if now - self.yawn_start_time >= self.YAWN_DURATION:
                            self.yawn_count += 1
                            self.yawn_counted = True
                else:
                    mouth_state = "嘴巴闭合"
                    self.in_yawn = False
                    self.yawn_counted = False
                    self.yawn_start_time = 0

                self.cached_mouth_state = mouth_state
                self.last_mouth_update_time = now
                self.mouth_state_lab.config(text=mouth_state)
                self.yawn_cnt_lab.config(text=f"哈欠次数：{self.yawn_count}")
        else:
            eye_state = "未检测"
            self.cached_mouth_state = "未检测"
            self.cached_mar = 0.0
            self.eye_close_start_time = 0
            self.long_eye_close = False

        valid_rec = []
        max_in_win = 0.0
        for end_t, dur in self.close_record:
            if now - end_t <= self.TIME_WINDOW:
                valid_rec.append((end_t, dur))
                if dur > max_in_win:
                    max_in_win = dur
        self.close_record = valid_rec
        self.max_eye_close_lab.config(text=f"一分钟内最长闭眼：{max_in_win:.2f}s")

        self.blink_times = [t for t in self.blink_times if now - t <= 60.0]
        blink_freq = len(self.blink_times)

        self.eye_state_lab.config(text=eye_state)
        self.blink_cnt_lab.config(text=f"眨眼次数：{self.blink_count}")
        self.blink_freq_lab.config(text=f"眨眼频率(次/分钟)：{blink_freq}")

        self.update_status_line(eye_state, mouth_state, now, blink_freq, behaviors)
        self.show_frame(img_draw)
        self.root.after(15, self.update_stream_frame)

    def update_status_line(self, eye_state, mouth_state, now_time, blink_freq, behaviors):
        max_in_win = 0.0
        for end_t, dur in self.close_record:
            if now_time - end_t <= self.TIME_WINDOW:
                if dur > max_in_win:
                    max_in_win = dur

        if max_in_win > self.LONG_CLOSE_THRESH:
            level_text = "重度疲劳！立即休息"
            level_color = "red"
        elif self.yawn_count > self.YAWN_LIGHT and blink_freq > self.BLINK_FREQ_THRESH:
            level_text = "中度疲劳，请休息"
            level_color = "gold"
        elif self.yawn_count > self.YAWN_LIGHT:
            level_text = "轻度疲劳，注意休息"
            level_color = "blue"
        else:
            level_text = "驾驶员清醒"
            level_color = "black"
        self.level_lab.config(text=level_text, fg=level_color)

        state_text = "行为正常"
        if len(behaviors) > 0:
            state_text = "检测到分心行为"
        elif self.long_eye_close:
            state_text = "请睁眼"
        elif "张开" in mouth_state and self.in_yawn:
            state_text = "正在打哈欠"

        if state_text == "行为正常":
            state_color = "black"
        else:
            state_color = "orange"
        self.state_lab.config(text=state_text, fg=state_color)

    def show_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        max_w, max_h = 760, 580
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        x = (760 - nw) // 2
        y = (580 - nh) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedMonitorGUI(root)
    root.mainloop()