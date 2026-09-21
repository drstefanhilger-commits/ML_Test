import os
import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier

from app.train_bpp.bpp_dataset import load_dataset_from_dir

# Pfade zu echten Drohnen / No-Drone (48k, wie bisher)
DRONE_TRAIN_DIR = "./data/train_48k/drone_real_train"
NO_DRONE_TRAIN_DIR = "./data/train_48k/no_drone_train"

DRONE_TEST_DIR = "./data/train_48k/drone_real_test"
NO_DRONE_TEST_DIR = "./data/train_48k/no_drone_test"

MODEL_OUT = "./models/bpp/uav_bandpass_model.joblib"


def ensure_model_dir():
    d = os.path.dirname(MODEL_OUT)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    print("[INFO] Lade Trainingsdaten (Bandpass-Pipeline)…")
    X_drone_train = load_dataset_from_dir(DRONE_TRAIN_DIR)
    X_no_train = load_dataset_from_dir(NO_DRONE_TRAIN_DIR)

    y_drone_train = np.ones(len(X_drone_train), dtype=int)
    y_no_train = np.zeros(len(X_no_train), dtype=int)

    X_train = np.vstack([X_drone_train, X_no_train])
    y_train = np.concatenate([y_drone_train, y_no_train])

    print(f"[INFO] Trainingsdaten: X={X_train.shape}, y={y_train.shape}")

    print("[INFO] Trainiere GradientBoostingClassifier (BPP)…")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("[INFO] Lade Testdaten (Bandpass-Pipeline)…")
    X_drone_test = load_dataset_from_dir(DRONE_TEST_DIR)
    X_no_test = load_dataset_from_dir(NO_DRONE_TEST_DIR)

    y_drone_test = np.ones(len(X_drone_test), dtype=int)
    y_no_test = np.zeros(len(X_no_test), dtype=int)

    X_test = np.vstack([X_drone_test, X_no_test])
    y_test = np.concatenate([y_drone_test, y_no_test])

    acc = clf.score(X_test, y_test)
    print(f"[RESULT] Test-Accuracy (BPP): {acc:.3f}")

    ensure_model_dir()
    dump(clf, MODEL_OUT)
    print(f"[INFO] Modell gespeichert unter: {MODEL_OUT}")


if __name__ == "__main__":
    main()
