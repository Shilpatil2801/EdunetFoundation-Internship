# 🇮🇳 EdunetFoundation-Internship  
## Indian Sign Language Recognition System

**Real-time ISL Alphabet & Digit Recognition using MediaPipe + LSTM**

---

## 📌 Project Overview

This project implements a **real-time Indian Sign Language (ISL) recognition system** capable of identifying **static alphabets (A–Z)** and **digits (0–9)** using:

- **MediaPipe Hand Landmarks** for feature extraction  
- **LSTM neural network** for temporal sequence learning  
- **Gradio** for interactive web-based real-time inference  
- **Hugging Face Spaces** for deployment  

The system works with **live webcam input** and provides **on-screen landmark visualization and prediction confidence**.

---

## 🎯 Key Features

- 🔴 Real-time webcam-based recognition  
- ✋ Hand landmark detection (21 points × 3D)  
- 🧠 LSTM-based deep learning model  
- 📊 Confidence-based prediction filtering  
- 🎥 Live landmark overlay  
- 🌐 Deployed on Hugging Face Spaces  
- 🧪 Custom dataset (A–Z, 0–9)  

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|-----------|
| Language | Python |
| Hand Tracking | MediaPipe |
| Deep Learning | TensorFlow / Keras |
| Model | LSTM |
| UI | Gradio |
| Deployment | Hugging Face Spaces |
| Data Format | `.npy` landmark sequences |

---

## 📂 Project Structure

```text
internshipproject/
│
├── app.py                  # Gradio application (real-time inference)
├── camera.py               # MediaPipe hand landmark extraction
├── inference_utils.py      # Model loading & prediction logic
├── isl_model.h5            # Trained LSTM model (legacy)
├── isl_model.keras         # Updated Keras model
├── class_names.npy         # Label mapping (A–Z, 0–9)
├── collectdata.py          # Dataset generation script
├── isl.py                  # Training / experimentation script
├── dataset/                # Custom landmark dataset
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitignore
