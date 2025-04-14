
import cv2 as cv
import mediapipe as mp
import numpy as np

# Инициализация mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Настройки для трекинга лица
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Настройки для трекинга позы
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 24) # Частота кадров
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600) # Ширина кадров в видеопотоке.
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480) # Высота кадров в видеопотоке.

def is_muscle_tensed(landmarks):
    # Пример логики для определения напряжения бицепса
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    left_distance = np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - np.array([left_elbow.x, left_elbow.y]))
    right_distance = np.linalg.norm(np.array([right_shoulder.x, right_shoulder.y]) - np.array([right_elbow.x, right_elbow.y]))

    # Если расстояние меньше определенного порога, считаем мышцу напряженной
    return left_distance < 0.1 or right_distance < 0.1

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Преобразование цветового пространства
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Трекинг лица
    face_results = face_mesh.process(img_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    # Трекинг позы
    pose_results = pose.process(img_rgb)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        if is_muscle_tensed(landmarks):
            connection_color = (0, 0, 255)  # Красный цвет для напряженных мышц
        else:
            connection_color = (255, 0, 0)  # Синий цвет для обычного состояния

        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=pose_results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2)
        )

    cv.imshow("camera", img)
    if cv.waitKey(10) == 27: # Клавиша Esc
        break

cap.release()
cv.destroyAllWindows()
