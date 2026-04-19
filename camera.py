import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # <--- CHANGED: Set to 2 to allow both hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

    # ---------------- FIX START ----------------
    # Case 1: 1 hand → pad to 126
    if len(keypoints) == 63:
        keypoints.extend([0] * 63)

    # Case 2: no hands → full zero vector
    elif len(keypoints) == 0:
        keypoints = [0] * 126
    # ---------------- FIX END ----------------

    return frame, keypoints