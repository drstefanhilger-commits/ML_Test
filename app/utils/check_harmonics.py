import os
import sys
import numpy as np

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train_hbo"
)
sys.path.append(BASE_DIR)

from utils_audio import load_audio
from features_harmonic import harmonic_features

base = "./data/train_48k/drone_real_train"
sr = 48000

energies = []

for root, _, files in os.walk(base):
    for f in files:
        if f.lower().endswith(".wav"):
            path = os.path.join(root, f)
            y = load_audio(path, sr)
            feat = harmonic_features(y, sr, 2048, 512, 6)
            energies.append(feat[-6:])  # die 6 harmonischen Energien

energies = np.array(energies)

print("\nDurchschnittliche harmonische Energien:")
print(np.mean(energies, axis=0))

print("\nMin/Max pro Harmonischer:")
print("Min:", np.min(energies, axis=0))
print("Max:", np.max(energies, axis=0))
