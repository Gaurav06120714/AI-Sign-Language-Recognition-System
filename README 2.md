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

## 🚀 Improvements

Based on a comprehensive analysis of the project, here are the key improvements needed, organized step-by-step from critical fixes to advanced enhancements.

### **PHASE 1: CRITICAL FIXES (Priority: Must Complete First)**

#### **Step 1: Complete the Dataset Collector**
**Issue:** The `dataset_collector.py` file is incomplete - the main loop cuts off mid-implementation.
**Fix:** 
- Complete the keyboard event handling for a-z keys
- Add frame counter and sample validation
- Implement proper cleanup on exit
- Add progress display showing samples collected per gesture

#### **Step 2: Implement the FastAPI Server**
**Issue:** `backend/api/main.py` is completely empty, but requirements.txt includes FastAPI.
**Fix:**
- Create FastAPI application with endpoints:
  - `POST /predict` for single gesture prediction
  - `WebSocket /stream` for real-time video processing
  - `GET /health` for system status
- Add proper CORS middleware for frontend integration

#### **Step 3: Build the React Frontend**
**Issue:** All frontend files are empty, no React app exists.
**Fix:**
- Initialize React app with Vite
- Create `package.json` with dependencies (React, WebSocket client)
- Implement core components: CameraFeed, GestureDisplay, SentenceOutput
- Add WebSocket connection to backend API

#### **Step 4: Fix Model Loading Issues**
**Issue:** Hard-coded paths and fragile error handling in `model_predictor.py`.
**Fix:**
- Use environment variables for model paths
- Add proper exception handling with custom error classes
- Validate model file existence and format before loading
- Return structured error responses instead of string messages

#### **Step 5: Add Input Validation**
**Issue:** No validation of feature vectors, landmarks, or inputs throughout the codebase.
**Fix:**
- Validate feature arrays are exactly 63 elements
- Check landmark coordinates are in valid ranges [0,1]
- Add bounds checking for all inputs
- Return meaningful error messages for invalid data

#### **Step 6: Clean Up Empty Modules**
**Issue:** Many files are empty or contain only whitespace, creating confusion.
**Fix:**
- Either implement or remove: `nlp/grammar_corrector.py`, `nlp/sentence_builder.py`, `speech/text_to_speech.py`, `utils/gesture_filter.py`
- Move TTS logic from `gesture_pipeline.py` to dedicated speech module
- Remove duplicate `scripts/train_model.py` if redundant

### **PHASE 2: ARCHITECTURE IMPROVEMENTS**

#### **Step 7: Refactor Monolithic Pipeline**
**Issue:** `gesture_pipeline.py` mixes vision, ML, TTS, and UI concerns.
**Fix:**
- Create separate services: GestureDetectionService, SentenceBuilderService, TextToSpeechService
- Implement dependency injection pattern
- Make components testable and reusable independently

#### **Step 8: Add Configuration Management**
**Issue:** Hard-coded values scattered throughout code (thresholds, paths, parameters).
**Fix:**
- Create `config/` directory with YAML configuration files
- Move all constants to configuration system
- Support different environments (dev/prod)

#### **Step 9: Implement Proper Error Handling**
**Issue:** Inconsistent error handling - some return strings, others return None.
**Fix:**
- Create custom exception classes (GestureError, ModelNotFoundError, etc.)
- Add try/catch blocks with proper logging
- Implement graceful degradation (fallbacks when components fail)

#### **Step 10: Add Type Hints and Documentation**
**Issue:** No type hints, minimal docstrings, making code hard to maintain.
**Fix:**
- Add type hints to all functions (e.g., `def extract_features(hand_landmarks) -> Optional[np.ndarray]`)
- Write comprehensive docstrings with parameters, returns, and examples
- Add inline comments for complex logic

### **PHASE 3: QUALITY ASSURANCE**

#### **Step 11: Implement Logging System**
**Issue:** No logging anywhere, impossible to debug issues.
**Fix:**
- Add logging throughout all modules
- Log key events: model loading, predictions, errors
- Configure different log levels for dev/prod

#### **Step 12: Create Unit Tests**
**Issue:** No tests exist, changes risk breaking functionality.
**Fix:**
- Create `tests/` directory with pytest
- Test core functions: feature extraction, model prediction
- Add mock data and fixtures for testing without hardware

#### **Step 13: Add Performance Monitoring**
**Issue:** No tracking of frame rates, latency, or resource usage.
**Fix:**
- Add FPS counter and timing measurements
- Profile model inference time
- Monitor memory and CPU usage

#### **Step 14: Improve Model Training**
**Issue:** Basic training with no validation, metrics, or optimization.
**Fix:**
- Add cross-validation and comprehensive metrics
- Implement hyperparameter tuning with GridSearchCV
- Save model metadata (accuracy, training date, parameters)
- Add feature importance analysis

### **PHASE 4: FEATURE COMPLETENESS**

#### **Step 15: Extend Gesture Support**
**Issue:** Only A-Z alphabet supported, no numbers or commands.
**Fix:**
- Add numbers 0-9 to dataset collection
- Include common words and commands (HELP, STOP, etc.)
- Support two-hand gestures for more complex signs

#### **Step 16: Add Confidence Scoring**
**Issue:** Model returns only labels, no confidence information.
**Fix:**
- Return prediction probability alongside label
- Add confidence threshold configuration
- Filter out low-confidence predictions

#### **Step 17: Implement NLP Features**
**Issue:** Grammar correction and sentence building modules are empty.
**Fix:**
- Implement grammar correction using spaCy
- Add sentence building logic with context awareness
- Support multiple languages

#### **Step 18: Enhance Text-to-Speech**
**Issue:** Basic TTS with no voice selection or quality options.
**Fix:**
- Add voice selection and speed control
- Implement offline/online TTS fallback
- Add speech queue management for smooth output

### **PHASE 5: DEPLOYMENT & DOCUMENTATION**

#### **Step 19: Create Docker Setup**
**Issue:** No containerization for deployment.
**Fix:**
- Create Dockerfile for backend
- Add docker-compose.yml for full stack
- Include model and dataset mounting

#### **Step 20: Complete Documentation**
**Issue:** README is outdated and incomplete.
**Fix:**
- Update README with installation, usage, API docs
- Create architecture diagrams
- Add troubleshooting guide and examples
- Document all configuration options

#### **Step 21: Add CI/CD Pipeline**
**Issue:** No automated testing or deployment.
**Fix:**
- Create GitHub Actions workflow
- Run tests on every push
- Add code quality checks (linting, type checking)
- Automate deployment to staging/production

#### **Step 22: Implement Security Measures**
**Issue:** No input validation, rate limiting, or security considerations.
**Fix:**
- Add request validation with Pydantic
- Implement rate limiting on API endpoints
- Add authentication if needed
- Sanitize all inputs

### **QUICK WINS (Can Fix in 1-2 Hours Each)**

1. **Add .gitignore** - Prevent committing models, datasets, cache files
2. **Pin dependency versions** - Update requirements.txt with specific versions
3. **Extract magic numbers** - Move hard-coded values to constants file
4. **Add environment variables** - Create .env file for configuration
5. **Fix dataset paths** - Consolidate to single canonical location

### **RECOMMENDED IMPLEMENTATION ORDER**

**Week 1:** Focus on Steps 1-6 (get basic functionality working)  
**Week 2:** Steps 7-10 (architectural improvements)  
**Week 3:** Steps 11-14 (testing and model improvements)  
**Week 4:** Steps 15-18 (feature completeness)  
**Week 5:** Steps 19-22 (deployment and polish)

---

**Made with ❤️ for accessible communication**