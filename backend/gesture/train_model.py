import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

BASE = os.path.dirname(__file__)
DATASET_FILE = os.path.abspath(os.path.join(BASE, os.getenv("DATA_PATH", "../../data/gesture_dataset.csv")))
MODEL_DIR    = os.path.abspath(os.path.join(BASE, "../models"))
MODEL_FILE   = os.path.join(MODEL_DIR, "gesture_model.pkl")
META_FILE    = os.path.join(MODEL_DIR, "model_meta.json")


def main():
    if not os.path.isfile(DATASET_FILE):
        print(f"Dataset not found at {DATASET_FILE}. Run dataset_collector.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE)
    if df.empty:
        print("Dataset is empty.")
        return

    print(f"Samples: {len(df)}  |  Gestures: {df['label'].nunique()}")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    # 5-fold cross-validation
    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "test_accuracy": round(test_accuracy, 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "total_samples": len(df),
        "gestures": sorted(df["label"].unique().tolist()),
        "model": "RandomForestClassifier",
        "n_estimators": 200,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved → {MODEL_FILE}")
    print(f"Metadata saved → {META_FILE}")


if __name__ == "__main__":
    main()
