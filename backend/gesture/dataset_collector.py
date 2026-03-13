import cv2
import os
import csv
import numpy as np
from hand_detector import HandDetector
from feature_extractor import extract_features

# Configuration
DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "gesture_dataset.csv")

def create_alphabet_mapping():
    """
    Create automatic key-to-label mapping for letters A-Z.
    Maps keyboard keys 'a' through 'z' to labels 'A' through 'Z'.
    """
    alphabet_map = {}
    for i in range(26):
        key_code = ord('a') + i  # ASCII codes for 'a' to 'z'
        label = chr(ord('A') + i)  # Labels 'A' to 'Z'
        alphabet_map[key_code] = label
    return alphabet_map

def initialize_dataset_file():
    """
    Create dataset directory and initialize CSV file with header if it doesn't exist.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    if not os.path.isfile(DATASET_FILE):
        with open(DATASET_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(1, 64)]
            writer.writerow(header)
        print(f"Created new dataset file: {DATASET_FILE}")
    else:
        print(f"Using existing dataset file: {DATASET_FILE}")

def save_sample(label, features):
    """
    Save a single sample (label + 63 features) to the CSV file.
    
    Args:
        label: The gesture label (A-Z)
        features: NumPy array of 63 feature values
    """
    row = [label] + features.tolist()
    with open(DATASET_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    # Initialize dataset
    initialize_dataset_file()
    
    # Create A-Z mapping
    LABEL_MAP = create_alphabet_mapping()
    
    # Initialize camera and hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    print("\n" + "="*60)
    print("    SIGN LANGUAGE ALPHABET DATA COLLECTOR (A-Z)")
    print("="*60)
    print("Instructions:")
    print("  • Press 'a' to record gesture A")
    print("  • Press 'b' to record gesture B")
    print("  • ... and so on up to 'z' for gesture Z")
    print("  • Press 'q' to quit")
    print("="*60 + "\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame from webcam.")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hands
        hand_landmarks_list = detector.detect_hands(frame)

        # Draw hand landmarks if detected
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                detector.draw_landmarks(frame, hand_landmarks)
        
        # Display instructions on frame
        cv2.putText(frame, "Press a-z to record letter | q to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display hand detection status
        status_text = f"Hand Detected: {'YES' if hand_landmarks_list else 'NO'}"
        status_color = (0, 255, 0) if hand_landmarks_list else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.imshow("Sign Language Alphabet Collector", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            print("\nExiting dataset collector...")
            break
        elif key in LABEL_MAP:
            label = LABEL_MAP[key]
            
            if hand_landmarks_list:
                # Extract features from the first detected hand
                features = extract_features(hand_landmarks_list[0])
                
                if features is not None and len(features) == 63:
                    # Save to CSV
                    save_sample(label, features)
                    print(f"✓ Recorded sample for: {label}")
                else:
                    print(f"✗ Error: Invalid feature vector for {label}")
            else:
                print(f"✗ No hand detected. Please show your hand for gesture {label}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDataset collection completed.")
    print(f"Data saved to: {DATASET_FILE}\n")

if __name__ == "__main__":
    main()