import gradio as gr
import cv2
import numpy as np
from camera import extract_keypoints
from inference_utils import predict_sign

SEQUENCE_LENGTH = 30

def recognize_sign(video_path):
    if video_path is None:
        return "No video received"

    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, keypoints = extract_keypoints(frame)

        # Normalize to 126 keypoints
        if len(keypoints) == 63:
            keypoints = np.concatenate([keypoints, np.zeros(63)])
        elif len(keypoints) != 126:
            keypoints = np.zeros(126)

        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

    cap.release()

    label, confidence = predict_sign(sequence)

    if label:
        return f"Prediction: {label}\nConfidence: {confidence:.2f}"
    else:
        return "Gesture not recognized confidently."

iface = gr.Interface(
    fn=recognize_sign,
    inputs=gr.Video(
        sources=["webcam", "upload"],
        label="Record or Upload ISL Gesture"
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Indian Sign Language Recognition System",
    description="LSTM-based ISL recognition using MediaPipe hand landmarks"
)

if __name__ == "__main__":
    iface.launch()
