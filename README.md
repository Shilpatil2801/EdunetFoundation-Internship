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
```
## 🧠 Model Details

- **Input shape:** `(30, 126)`
  - 30 frames  
  - 21 landmarks × (x, y, z) × 2 hands  

- **Architecture:**
  - LSTM layers  
  - Dense output layer with Softmax  

- **Loss function:** Categorical Crossentropy  
- **Output:** Alphabet or digit label with confidence score  

---

## 📊 Dataset Description

- Custom-collected dataset using webcam  
- Each class folder contains `.npy` files  
- Each file represents a **sequence of hand landmarks**

### Labels include:
- **Digits:** `0–9`  
- **Alphabets:** `A–Z`  

---

## 🚀 Running Locally

### 1️⃣ Create virtual environment
```bash
python -m venv isl_env
isl_env\Scripts\activate
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
### 3️⃣ Run the application
```bash
python app.py

## 🌐 Deployment (Hugging Face Spaces)

- **Framework:** Gradio  
- **Runtime:** CPU  
- **Webcam access:** Enabled  
- **Public demo:** Accessible via browser  

---

## 📷 Demo Capabilities

- Live webcam input  
- Real-time hand landmark visualization  
- Continuous prediction updates  
- Confidence thresholding to reduce false positives  

---

## ⚠️ Known Limitations

- Static gestures only (no continuous word recognition yet)  
- Sensitive to lighting and camera angle  
- Single-hand dominant gestures work best  

---

## 🔮 Future Enhancements

- ✅ Dynamic gesture recognition (words/sentences)  
- ✅ Temporal smoothing for stable predictions  
- ✅ Multi-hand gesture support  
- ✅ Transformer-based sequence models  
- ✅ Mobile-friendly deployment  

---

## 👩‍💻 Author

**Shilpa Patil**  
Artificial Intelligence & Data Science Student  
**Internship Project – ISL Recognition**
