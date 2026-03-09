import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from math import acos, degrees

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

# =========================
# Normalizar landmarks
# =========================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    # Usar muñeca como origen
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Escalar por tamaño de la mano
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks = landmarks / max_dist

    return landmarks.flatten()

# =========================
# Calcular ángulo entre 3 puntos
# =========================
def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return degrees(acos(np.clip(cos_angle, -1.0, 1.0)))

# =========================
# Ángulos de los dedos
# =========================
def compute_angles(landmarks):
    l = np.array(landmarks)

    angles = []

    # Índices MediaPipe
    fingers = {
        "thumb":  [1,2,3,4],
        "index":  [5,6,7,8],
        "middle": [9,10,11,12],
        "ring":   [13,14,15,16],
        "pinky":  [17,18,19,20]
    }

    for f in fingers.values():
        # Ángulo en las 3 articulaciones
        angles.append(angle(l[f[0]], l[f[1]], l[f[2]]))
        angles.append(angle(l[f[1]], l[f[2]], l[f[3]]))

    return angles

# =========================
# Procesar dataset
# =========================
dataset_path = "dataset"
rows = []

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

                norm = normalize_landmarks(landmarks)
                angles = compute_angles(landmarks)

                row = np.concatenate([norm, angles])
                rows.append(np.append(row, label))

# =========================
# Guardar CSV
# =========================
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]

for i in range(10):
    columns.append(f"angle{i}")

columns.append("label")

df = pd.DataFrame(rows, columns=columns)
df.to_csv("hand_dataset.csv", index=False)

print("Dataset generado:", df.shape)
