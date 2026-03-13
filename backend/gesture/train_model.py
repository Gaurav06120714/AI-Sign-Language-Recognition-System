import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Configuration
DATASET_FILE = "../../dataset/gesture_dataset.csv"
MODEL_DIR = "../../models"
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.pkl")

def main():
    if not os.path.isfile(DATASET_FILE):
        print(f"Dataset not found at {DATASET_FILE}. Please run dataset_collector.py first.")
        return

    # Load dataset CSV
    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE)

    if df.empty:
        print("Dataset is empty. Please collect some data first.")
        return

    print(f"Dataset size: {len(df)} samples")

    # Split into X (features) and y (labels)
    # The first column is 'label', the rest 63 columns are features
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train classifier using scikit-learn
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Calculate and print accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    # Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Trained model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
