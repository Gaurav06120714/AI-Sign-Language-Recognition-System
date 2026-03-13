import cv2
import time
import threading
from hand_detector import HandDetector
from feature_extractor import extract_features
import model_predictor

# Attempt to load pyttsx3 or gTTS
TTS_AVAILABLE = False
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    TTS_AVAILABLE = 'pyttsx3'
    
    def speak(text):
        # pyttsx3 blocking call runs in thread to prevent video freeze
        def _speak():
            tts_engine.say(text)
            tts_engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()
        
except ImportError:
    try:
        from gtts import gTTS
        import os
        from playsound import playsound
        TTS_AVAILABLE = 'gtts'
        
        def speak(text):
            def _speak():
                tts = gTTS(text=text, lang='en')
                temp_file = "temp_speech.mp3"
                tts.save(temp_file)
                playsound(temp_file)
                os.remove(temp_file)
            threading.Thread(target=_speak, daemon=True).start()
    except ImportError:
        def speak(text):
            pass

def main():
    # Capture webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize Hand Detector
    detector = HandDetector()

    # Sentence Builder Variables
    sentence = []
    current_prediction = None
    prediction_buffer_count = 0
    BUFFER_THRESHOLD = 10     # Needs N frames of consistent prediction to register the word
    SENTENCE_TIMEOUT = 3.0    # Seconds to wait before clearing/speaking sentence
    last_word_time = time.time()

    print("--- Sign Language Recognition Pipeline ---")
    if TTS_AVAILABLE:
        print(f"Text-to-Speech is ENABLED ({TTS_AVAILABLE}).")
    else:
        print("Text-to-Speech is DISABLED (Install pyttsx3 or gtts to enable).")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip horizontally for intuitive viewing
        frame = cv2.flip(frame, 1)

        # Detect hand landmarks
        hand_landmarks_list = detector.detect_hands(frame)
        gesture_label = "None"

        if hand_landmarks_list:
            # For simplicity, we process the first detected hand
            hand_landmarks = hand_landmarks_list[0]
            
            # Draw hand landmarks
            detector.draw_landmarks(frame, hand_landmarks)

            # Extract features
            features = extract_features(hand_landmarks)

            if features is not None:
                # Call predict_gesture()
                gesture_label = model_predictor.predict_gesture(features)

                # Sentence Builder Logic
                if gesture_label and gesture_label != "Model Not Found":
                    # Simple buffering logic to ensure stability
                    if gesture_label == current_prediction:
                        prediction_buffer_count += 1
                        if prediction_buffer_count == BUFFER_THRESHOLD:
                            # Avoid immediate duplicate words
                            if not sentence or sentence[-1] != gesture_label:
                                sentence.append(gesture_label)
                                last_word_time = time.time()
                    else:
                        current_prediction = gesture_label
                        prediction_buffer_count = 1
        else:
            current_prediction = None
            prediction_buffer_count = 0

        # Optional Text-to-Speech and automatic sentence clearing
        # We process 'thankyou' directly into 'thank you' if needed
        # We auto-speak and clear after SENTENCE_TIMEOUT seconds of inactivity
        if sentence and (time.time() - last_word_time > SENTENCE_TIMEOUT):
            text_to_speak = " ".join(sentence).replace("THANKYOU", "THANK YOU").capitalize()
            print(f"Speaking: {text_to_speak}")
            speak(text_to_speak)
            sentence = []

        # Display predicted gesture on screen
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        sentence_str = " ".join(sentence).replace("THANKYOU", "THANK YOU")
        if sentence_str:
            cv2.putText(frame, f"Sentence: {sentence_str}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        cv2.putText(frame, "Press 'c' to clear | 's' to manual speak | 'q' to quit",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Run in real time
        cv2.imshow("Sign Language System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
        elif key == ord('s') and sentence:
            text_to_speak = " ".join(sentence).replace("THANKYOU", "THANK YOU").capitalize()
            speak(text_to_speak)
            sentence = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
