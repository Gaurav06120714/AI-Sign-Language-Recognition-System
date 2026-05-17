import cv2
import os
import csv
import numpy as np
from dotenv import load_dotenv
from hand_detector import HandDetector
from feature_extractor import extract_features

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

BASE = os.path.dirname(__file__)
DATA_PATH = os.getenv("DATA_PATH", "../../data/gesture_dataset.csv")
DATASET_FILE = os.path.abspath(os.path.join(BASE, DATA_PATH))

LABEL_MAP = {ord('a') + i: chr(ord('A') + i) for i in range(26)}
SAMPLE_COUNTS: dict[str, int] = {}


def initialize_dataset():
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    if not os.path.isfile(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + [f"f{i}" for i in range(1, 64)])
        print(f"Created dataset: {DATASET_FILE}")
    else:
        import pandas as pd
        df = pd.read_csv(DATASET_FILE)
        for label, count in df["label"].value_counts().items():
            SAMPLE_COUNTS[label] = int(count)
        print(f"Existing dataset: {len(df)} samples across {df['label'].nunique()} gestures")


def save_sample(label: str, features: np.ndarray):
    with open(DATASET_FILE, "a", newline="") as f:
        csv.writer(f).writerow([label] + features.tolist())
    SAMPLE_COUNTS[label] = SAMPLE_COUNTS.get(label, 0) + 1


def main():
    initialize_dataset()

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    print("\n" + "=" * 55)
    print("   SIGN LANGUAGE DATA COLLECTOR  (A–Z)")
    print("=" * 55)
    print("  Press a–z to record that letter's gesture")
    print("  Press 'q' to quit")
    print("=" * 55 + "\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        hand_landmarks_list = detector.detect_hands(frame)

        if hand_landmarks_list:
            for hl in hand_landmarks_list:
                detector.draw_landmarks(frame, hl)

        detected = bool(hand_landmarks_list)
        status_color = (0, 220, 0) if detected else (0, 0, 220)
        cv2.putText(frame, f"Hand: {'DETECTED' if detected else 'NOT DETECTED'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "Press a-z to record | q to quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        counts_str = "  ".join([f"{k}:{v}" for k, v in sorted(SAMPLE_COUNTS.items())])
        cv2.putText(frame, counts_str[:80], (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        cv2.imshow("Sign Language Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in LABEL_MAP:  # BUG FIX: was ord('a') which quit on pressing A
            label = LABEL_MAP[key]
            if hand_landmarks_list:
                features = extract_features(hand_landmarks_list[0])
                if features is not None and len(features) == 63:
                    save_sample(label, features)
                    print(f"✓ {label}  (total: {SAMPLE_COUNTS[label]})")
                else:
                    print(f"✗ Bad features for {label}")
            else:
                print(f"✗ No hand detected for {label}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Dataset → {DATASET_FILE}")
    total = sum(SAMPLE_COUNTS.values())
    print(f"Total samples: {total}")


if __name__ == "__main__":
    main()
