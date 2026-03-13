import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """
        Detects hand landmarks from the input frame.
        """
        # Convert BGR to RGB as MediaPipe uses RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        return self.results.multi_hand_landmarks

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draws the 21 landmarks on the given frame.
        """
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS
        )
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame horizontally for intuitive mirror-like camera view
        frame = cv2.flip(frame, 1)

        # Detect hands
        hand_landmarks_list = detector.detect_hands(frame)

        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                # Draw landmarks
                frame = detector.draw_landmarks(frame, hand_landmarks)
        
        # Show video window
        cv2.imshow("Hand Detector", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()