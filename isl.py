import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset")
# -------------------------------------------

SEQUENCE_LENGTH = 30

# 1. Detect labels automatically
actions = np.array(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(actions)}

# Save class names ONCE (used by Streamlit & inference)
np.save(os.path.join(BASE_DIR, "class_names.npy"), actions)
print("Saved class names:", actions)

# 2. Load data
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for file_name in os.listdir(action_path):
        res = np.load(os.path.join(action_path, file_name))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

# 4. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

# 5. Compile & train
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=200,
    callbacks=[TensorBoard(log_dir=os.path.join(BASE_DIR, "logs"))]
)

# 6. Save model
model.save(os.path.join(BASE_DIR, "isl_model.h5"))
print("Model saved as isl_model.h5")
