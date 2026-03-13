import os
import pickle

# Configuration
# Resolving path relative to backend/gesture/
MODEL_FILE = "../../models/gesture_model.pkl"

class GesturePredictor:
    def __init__(self, model_path=MODEL_FILE):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")
        
        # Load trained model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, features):
        if features is None or len(features) != 63:
            return None
        
        # Input: NumPy array (63,)
        # Reshape to (1, 63) since sklearn expects a 2D array for a single sample
        input_data = features.reshape(1, -1)
        
        # Output: Predicted label
        prediction = self.model.predict(input_data)
        return prediction[0]

# Pre-load instance for standalone function usage
_predictor_instance = None

def predict_gesture(features):
    """
    Input: NumPy array (63,)
    Output: Predicted label
    """
    global _predictor_instance
    if _predictor_instance is None:
        try:
            _predictor_instance = GesturePredictor()
        except FileNotFoundError as e:
            print(e)
            return "Model Not Found"
            
    return _predictor_instance.predict(features)
