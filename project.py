import cv2
import mediapipe as mp
import numpy as np

# mediapipe 繪圖方法和樣式
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 計算角度函式
def calculate_angle(a, b, c):
    a = np.array(a)  # 第一點
    b = np.array(b)  # 第二點（角點）
    c = np.array(c)  # 第三點

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # 防止浮點數誤差
    return np.degrees(angle)

# 定義計算角度的關鍵點組合
angle_combinations = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_ELBOW', 'LEFT_WRIST', 'LEFT_THUMB'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_THUMB'),
    ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('LEFT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'LEFT_HIP'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_HAND'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_HAND'),
    ('LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_FOOT'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_FOOT'),
    ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_HIP')
]

cap = cv2.VideoCapture(0)

# 啟用姿勢偵測
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        img = cv2.resize(img, (720, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmarks_coords = {
                landmark.name: [landmarks[landmark.value].x, landmarks[landmark.value].y]
                for landmark in mp_pose.PoseLandmark
            }

            # 計算並顯示所有角度
            for point1, point2, point3 in angle_combinations:
                if point1 in landmarks_coords and point2 in landmarks_coords and point3 in landmarks_coords:
                    a = landmarks_coords[point1]
                    b = landmarks_coords[point2]
                    c = landmarks_coords[point3]
                    angle = calculate_angle(a, b, c)

                    # 在影像上顯示角度
                    if np.isfinite(angle):  # 確保角度是有效數值
                        angle_text = f'{int(angle)}'
                    else:
                        angle_text = 'Invalid'

                    pos = tuple(np.multiply(b, [720, 480]).astype(int))
                    # 嘗試使用簡單字體和不同顏色
                    cv2.putText(img, angle_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 繪製骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        # 水平翻轉畫面
       

        cv2.imshow('Pose Detection', img)
        if cv2.waitKey(5) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
