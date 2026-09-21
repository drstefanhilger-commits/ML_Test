import os
import sys
import numpy as np

# Pfad zu app/train_hbo bestimmen
BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train_hbo"
)
sys.path.append(BASE_DIR)

from utils_audio import load_audio
from features_harmonic import compute_f0_cepstrum

base = "./data/train_48k/drone_real_train"
sr = 48000

f0_values = []

for root, _, files in os.walk(base):
    for f in files:
        if f.lower().endswith(".wav"):
            path = os.path.join(root, f)
            try:
                y = load_audio(path, sr)
                f0 = compute_f0_cepstrum(y, sr)
                f0_values.append(f0)
            except Exception as e:
                print(f"Fehler bei Datei {path}: {e}")

print("\nAnzahl analysierter Dateien:", len(f0_values))

if len(f0_values) > 0:
    print("Min f0:", np.min(f0_values))
    print("Max f0:", np.max(f0_values))
    print("Durchschnitt f0:", np.mean(f0_values))

    hist, bins = np.histogram(f0_values, bins=20)
    print("\nHistogramm (f0-Verteilung):")
    for h, b in zip(hist, bins):
        print(f"{b:.1f} Hz: {h} Dateien")
