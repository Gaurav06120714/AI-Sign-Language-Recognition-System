import os
import pickle
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

MODEL_PATH = os.getenv("MODEL_PATH", "../models/gesture_model.pkl")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "../models/model_meta.json")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

_model = None
_classes = None


def _load_model():
    global _model, _classes
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_PATH))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Model not found at {abs_path}. Run train_model.py first.")
    with open(abs_path, "rb") as f:
        _model = pickle.load(f)
    _classes = list(_model.classes_)


def predict_gesture(features: np.ndarray) -> dict:
    """
    Returns {label, confidence, accepted} for a 63-element feature vector.
    """
    global _model, _classes
    if _model is None:
        _load_model()

    if features is None or len(features) != 63:
        return {"label": None, "confidence": 0.0, "accepted": False}

    proba = _model.predict_proba(features.reshape(1, -1))[0]
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])
    label = _classes[idx]
    accepted = confidence >= CONFIDENCE_THRESHOLD

    return {"label": label, "confidence": round(confidence, 4), "accepted": accepted}


def get_supported_gestures() -> list:
    global _model, _classes
    if _model is None:
        try:
            _load_model()
        except FileNotFoundError:
            return []
    return _classes
