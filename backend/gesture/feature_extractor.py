import numpy as np

def extract_features(hand_landmarks):
    """
    Extracts, normalizes, and flattens 21 hand landmarks into a 63-element vector.
    """
    if not hand_landmarks:
        return None

    # Step 1: Convert to NumPy array (21 x 3)
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Step 2: Translate wrist to origin
    # Landmark 0 is the wrist for MediaPipe
    base_point = landmarks[0]
    translated_landmarks = landmarks - base_point

    # Step 3: Scale to remove size differences
    max_value = np.max(np.abs(translated_landmarks))
    if max_value > 0:
        scaled_landmarks = translated_landmarks / max_value
    else:
        scaled_landmarks = translated_landmarks

    # Step 4: Flatten to 63-element vector
    flattened_features = scaled_landmarks.flatten()

    return flattened_features
