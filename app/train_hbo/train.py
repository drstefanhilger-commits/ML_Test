import os
import joblib
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from app.train_hbo.dataset import load_dataset

def main():
    with open("app/train_hbo/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sr = cfg["audio"]["sr"]
    n_fft = cfg["audio"]["n_fft"]
    hop = cfg["audio"]["hop_length"]
    n_harm = cfg["audio"]["n_harmonics"]

    # TRAIN
    X_drone_train, y_drone_train = load_dataset(cfg["paths"]["drone_train"], sr, n_fft, hop, n_harm)
    X_no_train, y_no_train = load_dataset(cfg["paths"]["no_drone_train"], sr, n_fft, hop, n_harm)

    X_train = np.vstack([X_drone_train, X_no_train])
    y_train = np.concatenate([y_drone_train, y_no_train])

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)

    # TEST
    X_drone_test, y_drone_test = load_dataset(cfg["paths"]["drone_test"], sr, n_fft, hop, n_harm)
    X_no_test, y_no_test = load_dataset(cfg["paths"]["no_drone_test"], sr, n_fft, hop, n_harm)

    X_test = np.vstack([X_drone_test, X_no_test])
    y_test = np.concatenate([y_drone_test, y_no_test])

    y_pred = clf.predict(X_test)

    print("[RESULT] Klassifikationsbericht:")
    print(classification_report(y_test, y_pred))

    print("[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # SAVE MODEL (Pfad-B-konform, deterministisch, sklearn-kompatibel)
    # ---------------------------------------------------------

    MODEL_DIR = os.path.join("models", "hbd")
    os.makedirs(MODEL_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_DIR, "uav_band_model.joblib")

    print(f"[INFO] Speichere Modell nach: {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    print("[INFO] Modell erfolgreich gespeichert.")

if __name__ == "__main__":
    main()
