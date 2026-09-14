#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic Feature Builder (Pfad‑B)
Generiert STM32‑kompatible 40‑Dim Feature‑Vektoren für:
    - drone_real
    - drone_sim
    - no_drone

Keine Hidden States, keine Randomness, reproduzierbar.
"""

import os
import numpy as np
import librosa
import json

# ------------------------------------------------------------
# 1. Deterministische Parameter
# ------------------------------------------------------------
SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

# ------------------------------------------------------------
# 2. Feature Extraktion (40‑Dim STM32‑kompatibel)
# ------------------------------------------------------------
def extract_features(path):
    """Extrahiert deterministische 40‑Dim Log‑Mel Features."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # Mel-Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    # Log-Mel
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Mittelwert über Zeit → 40‑Dim Vektor
    feat = np.mean(logmel, axis=1)

    return feat.astype(np.float32)

# ------------------------------------------------------------
# 3. Ordner definieren
# ------------------------------------------------------------
DATA_DIRS = {
    "drone_real": "data/train/drone_real",
    "drone_sim": "data/train/drone_sim",
    "no_drone": "data/train/no_drone"
}

OUT_DIR = "data/features"

# ------------------------------------------------------------
# 4. Deterministischer Build-Prozess
# ------------------------------------------------------------
def process_folder(name, in_dir):
    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".wav")])

    for f in files:
        in_path = os.path.join(in_dir, f)
        out_path = os.path.join(out_dir, f.replace(".wav", ".npy"))
        meta_path = os.path.join(out_dir, f.replace(".wav", ".json"))

        feat = extract_features(in_path)
        np.save(out_path, feat)

        meta = {
            "source": name,
            "file": f,
            "feature_dim": len(feat),
            "sample_rate": SAMPLE_RATE
        }
        with open(meta_path, "w") as fp:
            json.dump(meta, fp, indent=2)

        print(f"[OK] {name}: {f} → {out_path}")

# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
def main():
    print("======================================")
    print("     BUILD FEATURES (Pfad‑B V2)")
    print("======================================")

    for name, folder in DATA_DIRS.items():
        print(f"\n--- Processing: {name} ---")
        process_folder(name, folder)

    print("\n======================================")
    print("     FEATURES ERFOLGREICH ERSTELLT")
    print("======================================")

if __name__ == "__main__":
    main()
