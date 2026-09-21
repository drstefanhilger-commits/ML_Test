import os
import yaml
import numpy as np
import joblib
from tqdm import tqdm

from app.train_hbo.dataset import load_dataset


def main():
    # ---------------------------------------------------------
    # CONFIG LADEN
    # ---------------------------------------------------------
    with open("app/train_hbo/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sr = cfg["audio"]["sr"]
    n_fft = cfg["audio"]["n_fft"]
    hop = cfg["audio"]["hop_length"]
    n_harm = cfg["audio"]["n_harmonics"]

    SIM_PATH = cfg["paths"]["sim_drone_test"]

    # ---------------------------------------------------------
    # MODELL LADEN
    # ---------------------------------------------------------
    MODEL_PATH = "models/hbd/uav_band_model.joblib"
    print(f"[INFO] Lade Modell: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    print("[INFO] Modell erfolgreich geladen.")

    # ---------------------------------------------------------
    # SIMULIERTE DRONEN LADEN
    # ---------------------------------------------------------
    print(f"[INFO] Lade simulierte Drohnen aus {SIM_PATH}")
    X_sim, y_sim = load_dataset(SIM_PATH, sr, n_fft, hop, n_harm)

    print(f"[INFO] Anzahl Sim-Drohnen: {len(X_sim)}")

    # ---------------------------------------------------------
    # KLASSIFIKATION
    # ---------------------------------------------------------
    print("[INFO] Klassifiziere simulierte Drohnen...")
    y_pred = clf.predict(X_sim)
    y_prob = clf.predict_proba(X_sim)

    # ---------------------------------------------------------
    # AUSWERTUNG
    # ---------------------------------------------------------
    correct = np.sum(y_pred == y_sim)
    accuracy = correct / len(y_sim)

    print("\n=== SIM-DRONE TEST RESULT ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Correct: {correct}/{len(y_sim)}")

    # Einzelresultate
    print("\n=== Einzelklassifikation ===")
    for i in range(len(y_sim)):
        print(f"Sample {i:03d}: True={y_sim[i]}  Pred={y_pred[i]}  Prob={y_prob[i][1]:.3f}")

    print("\n[INFO] Test abgeschlossen.")


if __name__ == "__main__":
    main()
