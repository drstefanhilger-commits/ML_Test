import shutil
from pathlib import Path
import numpy as np
import soundfile as sf

from app.utils.sim_batch_generator import generate_simulation_batch
from app.train.train_pipeline import train_pipeline

# Ordner
DRONE_HARD = Path("data/train/drone_hard")
NO_DRONE_HARD = Path("data/train/no_drone_hard")
DRONE_SIM = Path("data/train/drone_sim")

TRUE_TEST_DIR = Path("test/true_test")
FALSE_TEST_DIR = Path("test/false_sounds")


# ---------------------------------------------------------
# Hard-Sample Finder
# ---------------------------------------------------------
def find_hard_samples(model, extract_features):
    """
    Findet automatisch:
      - False Negatives (echte Drohnen → NO DRONE)
      - False Positives (Umwelt → DRONE)
    und kopiert sie in die Hard-Ordner.
    """

    # Drohnen → FN
    for wav in TRUE_TEST_DIR.glob("*.wav"):
        feat = extract_features(wav)
        pred = model.predict(feat.reshape(1, -1))[0][0]
        if pred < 0.5:
            shutil.copy(wav, DRONE_HARD / wav.name)
            print(f"[FN] {wav.name} → kopiert nach drone_hard")

    # Umwelt → FP
    for wav in FALSE_TEST_DIR.glob("*.wav"):
        feat = extract_features(wav)
        pred = model.predict(feat.reshape(1, -1))[0][0]
        if pred >= 0.5:
            shutil.copy(wav, NO_DRONE_HARD / wav.name)
            print(f"[FP] {wav.name} → kopiert nach no_drone_hard")


# ---------------------------------------------------------
# Auto-Retraining Pipeline
# ---------------------------------------------------------
def auto_retrain():
    print("\n=== AUTO-RETRAINING START ===")

    # 1) Alte Simulationen löschen
    if DRONE_SIM.exists():
        shutil.rmtree(DRONE_SIM)
    DRONE_SIM.mkdir(parents=True, exist_ok=True)
    print("[OK] Alte Simulationen gelöscht")

    # 2) Neue Simulationen erzeugen
    print("[OK] Erzeuge neue Simulationen...")
    generate_simulation_batch(prefix="sim")

    # 3) Modell neu trainieren
    print("[OK] Starte Training...")
    train_pipeline()

    print("\n=== AUTO-RETRAINING DONE ===")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    auto_retrain()
