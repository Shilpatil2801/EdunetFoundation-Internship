import streamlit as st
import cv2
import numpy as np
from collections import deque

from camera import extract_keypoints
from inference_utils import predict_sign

st.set_page_config(page_title="ISL Recognition", layout="wide")

# ---------------- UI ----------------
st.title("🤟 Indian Sign Language Recognition")
st.markdown("Real-time gesture recognition using LSTM + MediaPipe")

run = st.sidebar.checkbox("Start Camera")

col1, col2 = st.columns(2)
frame_placeholder = col1.empty()
result_placeholder = col2.empty()

# ---------------- Buffer ----------------
sequence = deque(maxlen=30)

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    # Extract keypoints
    frame, keypoints = extract_keypoints(frame)

    if len(keypoints) > 0:
        sequence.append(keypoints)

    # Predict when sequence is full
    label, confidence = predict_sign(list(sequence))

    # Convert frame for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Display prediction
    if label:
        result_placeholder.markdown(
            f"""
            ## 🔤 Sign: **{label}**
            ### 🎯 Confidence: **{confidence:.2f}**
            """
        )
    else:
        result_placeholder.markdown("### Detecting...")

cap.release()