import csv
import numpy as np
import os
import random

# Define key landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

def create_base_hand():
    """Creates a base hand structure (flat open palm) centered at wrist (0,0,0) with appropriate scale."""
    landmarks = np.zeros((21, 3))

    # Wrist at 0,0,0

    # Fingers spread out roughly
    # Thumb
    landmarks[THUMB_CMC] = [-0.1, -0.05, 0.0]
    landmarks[THUMB_MCP] = [-0.2, -0.1, 0.0]
    landmarks[THUMB_IP] =  [-0.3, -0.15, 0.0]
    landmarks[THUMB_TIP] = [-0.4, -0.2, 0.0]

    # Index
    landmarks[INDEX_MCP] = [-0.1, -0.3, 0.0]
    landmarks[INDEX_PIP] = [-0.12, -0.45, 0.0]
    landmarks[INDEX_DIP] = [-0.13, -0.55, 0.0]
    landmarks[INDEX_TIP] = [-0.14, -0.65, 0.0]

    # Middle
    landmarks[MIDDLE_MCP] = [0.0, -0.3, 0.0] # Reference for scale! Dist to wrist ~ 0.3
    landmarks[MIDDLE_PIP] = [0.0, -0.48, 0.0]
    landmarks[MIDDLE_DIP] = [0.0, -0.6, 0.0]
    landmarks[MIDDLE_TIP] = [0.0, -0.7, 0.0]

    # Ring
    landmarks[RING_MCP] = [0.1, -0.28, 0.0]
    landmarks[RING_PIP] = [0.12, -0.44, 0.0]
    landmarks[RING_DIP] = [0.13, -0.54, 0.0]
    landmarks[RING_TIP] = [0.14, -0.63, 0.0]

    # Pinky
    landmarks[PINKY_MCP] = [0.2, -0.25, 0.0]
    landmarks[PINKY_PIP] = [0.23, -0.38, 0.0]
    landmarks[PINKY_DIP] = [0.25, -0.46, 0.0]
    landmarks[PINKY_TIP] = [0.27, -0.53, 0.0]

    return landmarks

def apply_finger_state(landmarks, finger_indices, state):
    """
    Modifies landmarks for a specific finger.
    state: 'OPEN', 'CLOSED', 'HOOK'
    finger_indices: [MCP, PIP, DIP, TIP]
    """
    mcp = landmarks[finger_indices[0]]
    pip = landmarks[finger_indices[1]]
    dip = landmarks[finger_indices[2]]
    tip = landmarks[finger_indices[3]]

    # Vector base (MCP to PIP usually defines direction)

    if state == 'CLOSED':
        # Fold finger into palm (y increases relative to MCP)
        # Simplify: just set positions relative to MCP
        landmarks[finger_indices[1]] = mcp + [0, 0.1, -0.1]
        landmarks[finger_indices[2]] = mcp + [0, 0.15, -0.15]
        landmarks[finger_indices[3]] = mcp + [0, 0.2, -0.05] # Tip touches palm

    elif state == 'OPEN':
        # Default is open, maybe add slight variation
        pass

def generate_sample(letter):
    landmarks = create_base_hand()

    # Define finger indices
    thumb = [1, 2, 3, 4]
    index = [5, 6, 7, 8]
    middle = [9, 10, 11, 12]
    ring = [13, 14, 15, 16]
    pinky = [17, 18, 19, 20]

    # Letters logic (approximate)
    if letter == 'A':
        # Fist with thumb on side
        apply_finger_state(landmarks, index, 'CLOSED')
        apply_finger_state(landmarks, middle, 'CLOSED')
        apply_finger_state(landmarks, ring, 'CLOSED')
        apply_finger_state(landmarks, pinky, 'CLOSED')
        # Thumb upright-ish against side
        landmarks[4] = landmarks[5] + [-0.05, -0.1, 0]

    elif letter == 'B':
        # Open palm, thumb tucked
        apply_finger_state(landmarks, index, 'OPEN')
        apply_finger_state(landmarks, middle, 'OPEN')
        apply_finger_state(landmarks, ring, 'OPEN')
        apply_finger_state(landmarks, pinky, 'OPEN')
        # Thumb crossed over palm
        landmarks[4] = landmarks[13] + [0, 0, -0.05]

    elif letter == 'C':
        # C shape
        # Curve all fingers
        pass # Hard to simulate accurately, leaving as open-ish but curved

    elif letter == 'V':
        apply_finger_state(landmarks, index, 'OPEN')
        apply_finger_state(landmarks, middle, 'OPEN')
        apply_finger_state(landmarks, ring, 'CLOSED')
        apply_finger_state(landmarks, pinky, 'CLOSED')

        # Spread index and middle
        landmarks[8] = landmarks[5] + [-0.1, -0.3, 0] # Index left
        landmarks[12] = landmarks[9] + [0.1, -0.3, 0]  # Middle right

    elif letter == 'L':
        apply_finger_state(landmarks, index, 'OPEN')
        apply_finger_state(landmarks, middle, 'CLOSED')
        apply_finger_state(landmarks, ring, 'CLOSED')
        apply_finger_state(landmarks, pinky, 'CLOSED')
        # Thumb out
        landmarks[4] = landmarks[2] + [-0.2, 0, 0]

    # Add noise
    noise = np.random.normal(0, 0.01, landmarks.shape)
    landmarks += noise

    return landmarks

def normalize_landmarks(landmarks):
    """Normalizes landmarks using the same logic as lsf_model.py"""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    scale = np.linalg.norm(wrist - middle_mcp)
    if scale == 0: scale = 1.0

    coords = []
    for lm in landmarks:
        # (x,y,z) relative to wrist, divided by scale
        norm_lm = (lm - wrist) / scale
        coords.extend(norm_lm)

    return coords

def generate_dataset(output_path, samples_per_class=100):
    letters = ['A', 'B', 'L', 'V'] # Only doing a subset that is easy to model procedurally

    data = []
    print(f"Generating {samples_per_class} samples for each of {letters}...")

    for letter in letters:
        for _ in range(samples_per_class):
            landmarks = generate_sample(letter)
            features = normalize_landmarks(landmarks)
            row = [letter] + features
            data.append(row)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Saved {len(data)} samples to {output_path}")

if __name__ == "__main__":
    generate_dataset("machine_learning/data.csv")
