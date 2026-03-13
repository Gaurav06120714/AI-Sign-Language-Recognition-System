# AI Sign Language Recognition System

## 📌 Overview

A real-time AI-powered Sign Language Recognition system that translates hand gestures into readable text and speech. This project uses **MediaPipe** for hand tracking, **OpenCV** for real-time video capture, and **Machine Learning (scikit-learn)** for gesture classification.

The system detects 21 hand landmarks, extracts numerical features, predicts the corresponding gesture using a trained model, and displays the result in real-time with optional text-to-speech output.

**Goal:** Bridge the communication gap between sign language users and non-sign language users by converting gestures into understandable text and speech.

---

## 🏗️ System Architecture

```
Camera Input
     ↓
OpenCV Frame Capture
     ↓
MediaPipe Hand Detection
     ↓
21 Hand Landmarks (x, y, z)
     ↓
Feature Extraction (63 Features)
     ↓
Gesture Classification Model
     ↓
Predicted Gesture Label
     ↓
Sentence Builder
     ↓
Text-to-Speech Output (Optional)
```

---

## 📁 Project Structure

```
sign-language/
│
├── backend/
│   └── gesture/
│       ├── hand_detector.py          # Hand detection using MediaPipe
│       ├── feature_extractor.py      # Extract 63 features from landmarks
│       ├── dataset_collector.py      # Collect training data
│       ├── train_model.py            # Train ML model
│       ├── model_predictor.py        # Load model and predict gestures
│       └── gesture_pipeline.py       # Real-time gesture recognition
│
├── dataset/
│   └── gesture_dataset.csv           # Training dataset (label, f1, f2, ..., f63)
│
├── models/
│   └── gesture_model.pkl             # Trained ML model
│
└── README.md
```

---

## 🧩 Core Components

### 1. **Hand Detection** (`hand_detector.py`)

Uses **MediaPipe Hands** to detect hands from webcam frames.

**Responsibilities:**
- Detect hand landmarks (21 points per hand)
- Return detected landmark coordinates
- Draw 21 landmark points on the frame

**Output:**  
21 hand landmarks with (x, y, z) coordinates

---

### 2. **Feature Extraction** (`feature_extractor.py`)

Converts hand landmarks into machine learning features.

**Process:**
1. Convert landmarks to NumPy array
2. Translate wrist landmark to origin (translation invariance)
3. Normalize scale (scale invariance)
4. Flatten into feature vector

**Output:**  
**63 Features** = 21 landmarks × (x, y, z)

```
[x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
```

This ensures:
- ✅ Translation invariance
- ✅ Scale invariance
- ✅ Consistent feature representation

---

### 3. **Dataset Collection** (`dataset_collector.py`)

Captures gesture samples for training.

**Workflow:**

```
Webcam
   ↓
Hand Detection
   ↓
Feature Extraction
   ↓
User presses key for gesture label
   ↓
Save features to dataset
```

**Dataset Format:**

```csv
label,f1,f2,f3,...,f63
A,0.23,0.12,...
B,0.41,0.18,...
```

**Supported Gestures:**
- **A-Z** (26 alphabet letters)
- **Words:** HI, BYE, YES, NO, THANKYOU, SORRY, PLEASE, HELP, STOP

---

### 4. **Model Training** (`train_model.py`)

Trains a gesture classification model using **Random Forest Classifier**.

**Steps:**

```
Load dataset
      ↓
Split features and labels
      ↓
Train ML model (RandomForestClassifier)
      ↓
Evaluate accuracy
      ↓
Save trained model → models/gesture_model.pkl
```

---

### 5. **Gesture Prediction** (`model_predictor.py`)

Loads the trained model and predicts gestures.

**Input:**  
63-element feature vector

**Output:**  
Predicted gesture label

```
[features] → model → "HELLO"
```

---

### 6. **Real-Time Gesture Pipeline** (`gesture_pipeline.py`)

Runs the full AI system in real-time.

**Pipeline:**

```
Camera
   ↓
Hand Detection
   ↓
Feature Extraction
   ↓
Gesture Prediction
   ↓
Display Result
```

**Display Example:**

```
Gesture: HELLO
Sentence: HELLO THANKYOU
```

---

## 🔤 Sentence Builder

Gestures are accumulated into a sentence buffer for meaningful communication.

**Example:**

```
HI + THANKYOU + HELP
        ↓
"HI THANKYOU HELP"
```

The buffer prevents noisy predictions by requiring multiple consistent frames before accepting a gesture.

---

## 🔊 Text-to-Speech (Optional)

The system can convert recognized gestures into speech using:
- **pyttsx3** (offline)
- **gTTS** (Google Text-to-Speech, requires internet)

**Example:**

```
HELLO THANKYOU
     ↓
🔊 "Hello, thank you"
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **OpenCV** | Real-time video capture and frame processing |
| **MediaPipe** | Hand landmark detection (21 points) |
| **NumPy** | Numerical feature representation |
| **scikit-learn** | Machine learning model (RandomForestClassifier) |
| **pandas** | Dataset handling and CSV operations |
| **pyttsx3** | Offline text-to-speech |
| **gTTS** | Online text-to-speech (Google) |
| **playsound** | Audio playback |

---

## ⚙️ Installation

### Prerequisites
- Python 3.7+
- Webcam

### Install Dependencies

```bash
pip install opencv-python mediapipe numpy scikit-learn pandas pyttsx3 gtts playsound
```

---

## 🚀 Usage

### Step 1: Collect Dataset

Run the dataset collector to capture gesture samples:

```bash
python backend/gesture/dataset_collector.py
```

**Instructions:**
- Press **a-z** to record alphabet gestures (A-Z)
- Press **q** to quit
- Ensure your hand is visible in the frame

The data will be saved to `dataset/gesture_dataset.csv`

---

### Step 2: Train the Model

Train the gesture recognition model:

```bash
python backend/gesture/train_model.py
```

This will:
- Load the dataset
- Train a RandomForestClassifier
- Evaluate accuracy
- Save the model to `models/gesture_model.pkl`

---

### Step 3: Run Real-Time Recognition

Start the real-time gesture recognition system:

```bash
python backend/gesture/gesture_pipeline.py
```

**Features:**
- Real-time hand detection
- Gesture prediction
- Sentence building
- Optional text-to-speech

---

## 📊 Model Performance

The model's accuracy depends on:
- Quality and quantity of training data
- Consistency of hand positions
- Lighting conditions

**Recommended:**
- Collect at least **50-100 samples per gesture**
- Use consistent lighting
- Maintain similar hand distances from the camera

---

## 🎯 Future Enhancements

- [ ] Support for two-hand gestures
- [ ] Deep learning model (CNN/LSTM)
- [ ] Mobile app deployment
- [ ] Real-time grammar correction
- [ ] Multi-language support
- [ ] Gesture autocomplete
- [ ] Integration with video conferencing platforms
- [ ] Gesture speed detection
- [ ] Context-aware predictions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Your Name**  
📧 Email: your.email@example.com  
🔗 GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **MediaPipe** for excellent hand tracking capabilities
- **OpenCV** for comprehensive computer vision tools
- **scikit-learn** for powerful machine learning algorithms
- The sign language community for inspiration and guidance

---

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Contact via email
- Check the documentation in each module

---

## 🔍 Troubleshooting

### Common Issues

**Issue:** Webcam not detected
- **Solution:** Check camera permissions and ensure no other application is using the webcam

**Issue:** Low prediction accuracy
- **Solution:** Collect more training samples (50-100 per gesture) with varied hand positions

**Issue:** Hand landmarks not detected
- **Solution:** Ensure good lighting and keep hand within camera frame

**Issue:** Model file not found
- **Solution:** Run `train_model.py` before running `gesture_pipeline.py`

---

## 📚 Additional Resources

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Sign Language Basics](https://www.startasl.com/)

---

**⭐ If you find this project helpful, please give it a star!**

---

**Made with ❤️ for accessible communication**