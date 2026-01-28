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
    """
    Processes a frame, extracts hand landmarks, and returns:
    1. The frame with landmarks drawn on it.
    2. A flattened list of keypoints:
       - 63 values if 1 hand is detected
       - 126 values if 2 hands are detected
       - Empty list [] if no hands are detected
    """
    # 1. Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Process the image
    results = hands.process(image_rgb)
    
    # 3. Draw landmarks and extract coordinates
    keypoints = [] 
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the original frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract (x, y, z) for every landmark
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
                
    # Note: We removed the slicing [:63] so it can now return 
    # all data (126 floats) if two hands are found.
        
    return frame, keypoints