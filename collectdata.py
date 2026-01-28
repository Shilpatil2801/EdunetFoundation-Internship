import cv2
import numpy as np
import os
from camera import extract_keypoints

# --- CONFIGURATION ---
LABEL = "A"            # CHANGE THIS PER RUN
SEQUENCE_LENGTH = 30   # frames per sample
NUM_SAMPLES = 50       # per word
DATA_PATH = "dataset"
# ---------------------

# Create directory for the label
os.makedirs(f"{DATA_PATH}/{LABEL}", exist_ok=True)

cap = cv2.VideoCapture(0)

sample_count = 0
sequence = []

print(f"Collecting data for: {LABEL}")
print("Press 'q' to quit manually.")

while sample_count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract keypoints from the current frame
    # (Assumes extract_keypoints returns a flattened list/array)
    frame, keypoints = extract_keypoints(frame)
    
    # Display the collection progress
    cv2.putText(frame, f"Collecting: {LABEL} ({sample_count}/{NUM_SAMPLES})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting ISL Data", frame)

    # --- LOGIC CHANGE START ---
    
    # CASE 1: One hand detected (21 landmarks * 3 coords = 63)
    if len(keypoints) == 63:
        # Pad with 63 zeros to make it compatible with 2-hand data
        padded_keypoints = np.concatenate([keypoints, np.zeros(63)])
        sequence.append(padded_keypoints)

    # CASE 2: Two hands detected (42 landmarks * 3 coords = 126)
    elif len(keypoints) == 126:
        # No padding needed
        sequence.append(keypoints)
        
    # --- LOGIC CHANGE END ---

    # Once we have 30 frames (SEQUENCE_LENGTH), save the file
    if len(sequence) == SEQUENCE_LENGTH:
        save_path = f"{DATA_PATH}/{LABEL}/{LABEL}_{sample_count}.npy"
        np.save(save_path, np.array(sequence))
        
        print(f"Saved sample {sample_count} to {save_path}")
        sample_count += 1
        sequence = [] # Reset for the next sample

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()