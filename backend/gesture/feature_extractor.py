import numpy as np


def extract_features(hand_landmarks):
    """Extract 63 normalized features from MediaPipe Python hand landmarks."""
    if not hand_landmarks:
        return None

    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    translated = landmarks - landmarks[0]
    max_val = np.max(np.abs(translated))
    scaled = translated / max_val if max_val > 0 else translated

    return scaled.flatten()


def extract_features_from_list(landmarks_list: list):
    """Extract 63 normalized features from MediaPipe JS landmark list [{x,y,z}...]."""
    if not landmarks_list or len(landmarks_list) != 21:
        return None

    landmarks = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks_list])
    translated = landmarks - landmarks[0]
    max_val = np.max(np.abs(translated))
    scaled = translated / max_val if max_val > 0 else translated

    return scaled.flatten()
