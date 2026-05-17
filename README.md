# 🤟 AI Sign Language Recognition

Real-time hand gesture recognition that translates sign language into text and speech — runs in the browser using MediaPipe JS + FastAPI WebSocket backend.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Hand Detection | MediaPipe Hands (JavaScript — runs in browser) |
| Backend API | FastAPI + WebSocket |
| ML Model | scikit-learn RandomForestClassifier |
| Frontend | React + Tailwind CSS |
| Text-to-Speech | Web Speech API (browser native) |
| Config | python-dotenv + Vite .env |

---

## How It Works

```
Browser webcam
      ↓
MediaPipe JS → 21 hand landmarks (x,y,z)
      ↓
Feature extraction (63 values) → WebSocket → FastAPI
      ↓
RandomForest predict_proba() → {label, confidence}
      ↓
React UI → display gesture + confidence bar
      ↓
Web Speech API → speak sentence
```

---

## Project Structure

```
ai-sign-language/
├── backend/
│   ├── api/
│   │   └── main.py              ← FastAPI WebSocket server
│   ├── gesture/
│   │   ├── hand_detector.py     ← MediaPipe Python (for data collection)
│   │   ├── feature_extractor.py ← 63-feature normalization
│   │   ├── dataset_collector.py ← collect training data via webcam
│   │   ├── train_model.py       ← train with 5-fold cross-validation
│   │   └── model_predictor.py   ← predict with confidence scores
│   ├── models/
│   │   ├── gesture_model.pkl    ← trained model (generated after training)
│   │   └── model_meta.json      ← accuracy, gestures list, date
│   ├── .env                     ← MODEL_PATH, DATA_PATH, PORT
│   └── requirements.txt
├── data/
│   └── gesture_dataset.csv      ← dataset (A-Z gestures, 60 samples each)
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── CameraFeed.jsx      ← MediaPipe JS webcam + skeleton
│   │   │   ├── GestureDisplay.jsx  ← gesture + confidence bar
│   │   │   ├── DetectedWords.jsx   ← word history chips
│   │   │   └── SentenceOutput.jsx  ← sentence + speak/copy/clear
│   │   └── services/
│   │       └── websocket.js        ← WS connection to FastAPI
│   ├── .env                        ← VITE_WS_URL
│   └── package.json
└── README.md
```

---

## Full Setup & Run Guide

### Step 1 — Clone the repo

```bash
git clone https://github.com/Gaurav06120714/ai-sign-language.git
cd ai-sign-language
```

---

### Step 2 — Create Python virtual environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> Note: Use `python3` not `python` on macOS

---

### Step 4 — Collect real training data (optional)

The repo includes sample data (`data/gesture_dataset.csv`) with 60 samples per letter.
To collect your own real hand data:

```bash
cd backend/gesture
python3 dataset_collector.py
```

- Press `a` to `z` to record each gesture (aim for 50+ samples per letter)
- Press `q` to quit
- Data saves automatically to `data/gesture_dataset.csv`

---

### Step 5 — Train the model

```bash
cd backend/gesture
python3 train_model.py
```

Output example:
```
Samples: 1560  |  Gestures: 26
Running 5-fold cross-validation...
CV Accuracy: 94.23% ± 1.12%
Test Accuracy: 95.10%
Model saved → backend/models/gesture_model.pkl
Metadata saved → backend/models/model_meta.json
```

---

### Step 6 — Start the FastAPI backend

```bash
cd backend/api
source ../venv/bin/activate
python3 main.py
```

Server runs at: **http://localhost:8000**

Check it's working:
```bash
curl http://localhost:8000/health
```

---

### Step 7 — Start the React frontend

Open a **new terminal tab**:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

### Step 8 — Use the app

1. Open **http://localhost:5173** in your browser
2. Allow camera access when prompted
3. Click **▶ Start**
4. Show hand gestures in front of the camera
5. Detected letters appear with a confidence bar
6. Words build into a sentence automatically
7. Click **🔊 Speak** to hear the sentence

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status + gesture count |
| `/gestures` | GET | List of all supported gestures |
| `/ws` | WebSocket | Send `{landmarks: [{x,y,z}×21]}` → receive `{label, confidence, accepted}` |

---

## Controls

| Button | Action |
|--------|--------|
| ▶ Start | Begin gesture detection |
| ⏸ Pause | Pause detection |
| 🔊 Speak | Read sentence aloud (Web Speech API) |
| Copy | Copy sentence to clipboard |
| Clear | Reset all words and sentence |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `zsh: command not found: python` | Use `python3` instead of `python` |
| `ModuleNotFoundError: No module named 'fastapi'` | Run `source backend/venv/bin/activate` first |
| Camera not detected | Allow camera permission in browser settings |
| Low accuracy | Collect more real data using `dataset_collector.py` |
| WebSocket not connecting | Make sure backend is running on port 8000 |
