import os
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "isl_model.h5"))
actions = np.load(os.path.join(BASE_DIR, "class_names.npy"))

SEQUENCE_LENGTH = 30
threshold = 0.8

def predict_sign(sequence):
    if len(sequence) < SEQUENCE_LENGTH:
        return None, None

    res = model.predict(
        np.expand_dims(sequence, axis=0),
        verbose=0
    )[0]

    idx = np.argmax(res)
    confidence = res[idx]

    if confidence > threshold:
        return actions[idx], confidence

    return None, None
