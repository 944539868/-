# model2.py
import cv2
import numpy as np
import dlib

class FatigueDetector:
    def __init__(self, predictor_path="weights/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.EYE_LEFT = list(range(36, 42))
        self.EYE_RIGHT = list(range(42, 48))
        self.MOUTH = [48, 50, 52, 54, 56, 58]

        self.EAR_NORMAL = 0.25
        self.EAR_HALF = 0.18
        self.MAR_NORMAL = 0.6
        self.MAR_SIDE = 0.72
        self.ANGLE_THRESH = 15

        # 俯仰角校准偏移（+30°）
        self.pitch_offset = 30.0

    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        if len(faces) == 0:
            return None
        shape = self.predictor(gray, faces[0])
        return np.array([[p.x, p.y] for p in shape.parts()])

    def calculate_ear(self, eye_points):
        p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
        p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
        p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def calculate_mar(self, mouth_points):
        p51_p59 = np.linalg.norm(mouth_points[1] - mouth_points[5])
        p53_p57 = np.linalg.norm(mouth_points[2] - mouth_points[4])
        p49_p55 = np.linalg.norm(mouth_points[0] - mouth_points[3])
        return (p51_p59 + p53_p57) / (2.0 * p49_p55)

    def get_face_angle(self, landmarks):
        """倾斜角 (roll)，基于两眼连线角度"""
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return abs(angle)

    def get_pitch_angle(self, landmarks, frame_shape=None):
        """
        俯仰角 (pitch)：
        用鼻尖(30)到两眼中心连线的垂直距离与眼间距的比值估算
        抬头为正，低头为负，范围约 -45° ~ +45°
        已加入 +30° 校准偏移
        """
        # 两眼中心
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        eye_center = (left_eye + right_eye) / 2.0

        # 鼻尖
        nose = landmarks[30]

        # 眼间距（像素）
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-6:
            return 0.0

        # 鼻尖到两眼中心的垂直距离（图像 Y 向下，若鼻尖高于眼中心，dy 为正）
        dy = eye_center[1] - nose[1]
        # 比例换算成角度，系数 50 可根据实际效果微调
        pitch_ratio = dy / eye_dist
        pitch_deg = pitch_ratio * 50.0

        # 加上校准偏移
        pitch_deg += self.pitch_offset

        # 限制范围到 ±45°（加上偏移后实际范围可能变成 -15° ~ 75°，根据你需求再调整）
        pitch_deg = np.clip(pitch_deg, -45, 45)

        return float(pitch_deg)

    def get_yaw_angle(self, landmarks, frame_shape=None):
        """
        水平偏转角 (yaw)：
        用鼻尖到两眼连线的水平偏移与眼间距的比例估算
        右转为正，左转为负，范围约 -45° ~ +45°
        """
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        eye_center = (left_eye + right_eye) / 2.0

        nose = landmarks[30]

        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-6:
            return 0.0

        # 鼻尖到两眼中心的水平偏移（正表示鼻尖偏右）
        dx = nose[0] - eye_center[0]
        yaw_ratio = dx / eye_dist
        yaw_deg = yaw_ratio * 50.0

        yaw_deg = np.clip(yaw_deg, -45, 45)

        return float(yaw_deg)