import cv2 as cv
import mediapipe as mp
import pygame
import numpy as np
import math

# Инициализация mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Настройки для трекинга лица и позы
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Инициализация Pygame
pygame.init()
width, height = 600, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Face and Pose Tracking")

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 24)  # Частота кадров
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)  # Ширина кадров в видеопотоке.
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # Высота кадров в видеопотоке.

def draw_landmarks(image, landmarks, connections, landmark_color=(255, 0, 0), connection_color=(0, 255, 0)):
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        pygame.draw.circle(screen, landmark_color, (x, y), 2)

    for connection in connections:
        x1 = int(landmarks[connection[0]].x * width)
        y1 = int(landmarks[connection[0]].y * height)
        x2 = int(landmarks[connection[1]].x * width)
        y2 = int(landmarks[connection[1]].y * height)
        pygame.draw.line(screen, connection_color, (x1, y1), (x2, y2), 2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Преобразование цвета из BGR в RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Обработка изображения для трекинга лица и позы
    face_results = face_mesh.process(img_rgb)
    pose_results = pose.process(img_rgb)

    # Очистка экрана
    screen.fill((0, 0, 0))

    # Рисование сетки лица
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            draw_landmarks(img, face_landmarks.landmark, mp_face_mesh.FACEMESH_CONTOURS)

    # Рисование скелета
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        connections = mp_pose.POSE_CONNECTIONS

        # Определение напряжения мышц
        left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
        right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

        left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

        # Определение цвета линий в зависимости от напряжения мышц
        left_arm_color = (255, 0, 0) if left_elbow_angle < 90 else (0, 255, 0)
        right_arm_color = (255, 0, 0) if right_elbow_angle < 90 else (0, 255, 0)
        left_leg_color = (255, 0, 0) if left_knee_angle < 90 else (0, 255, 0)
        right_leg_color = (255, 0, 0) if right_knee_angle < 90 else (0, 255, 0)

        # Рисование скелета с учетом напряжения мышц
        for connection in connections:
            if connection in [(11, 13), (13, 15), (12, 14), (14, 16)]:
                color = left_arm_color if connection in [(11, 13), (13, 15)] else right_arm_color
            elif connection in [(23, 25), (25, 27), (24, 26), (26, 28)]:
                color = left_leg_color if connection in [(23, 25), (25, 27)] else right_leg_color
            else:
                color = (0, 255, 0)

            x1 = int(landmarks[connection[0]].x * width)
            y1 = int(landmarks[connection[0]].y * height)
            x2 = int(landmarks[connection[1]].x * width)
            y2 = int(landmarks[connection[1]].y * height)
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)

    # Обновление экрана
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    if cv.waitKey(10) == 27:  # Клавиша Esc
        break

cap.release()
pygame.quit()
