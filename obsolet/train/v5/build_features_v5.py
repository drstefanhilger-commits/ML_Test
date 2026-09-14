#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD FEATURES V5 (GLOBAL NORMALIZATION)
- Extrahiert Features aus ALLEN WAVs:
    train_48k/drone_real
    train_48k/drone_sim
    train_48k/no_drone
    test/true_test
    test/false_test
- Berechnet globale Z‑Norm
- Speichert Scaler
- Speichert normalisierte Features für Training
"""

import os
import numpy as np
import librosa
import json
from pathlib import Path

SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

DATA_DIRS = {
    "drone_real": "data/train_48k/drone_real",
    "drone_sim":  "data/train_48k/drone_sim",
    "no_drone":   "data/train_48k/no_drone",
    "true_test":  "test/true_test",
    "false_test": "test/false_test"
}

OUT_DIR = Path("data/features_v5")
OUT_DIR.mkdir(exist_ok=True)

SCALER_JSON = Path("models/binary/scaler_v5.json")

def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return np.mean(logmel, axis=1).astype(np.float32)

def main():
    all_feats = []
    all_labels = []

    print("=== BUILD FEATURES V5 ===")

    for name, folder in DATA_DIRS.items():
        print(f"\n--- {name} ---")
        out_sub = OUT_DIR / name
        out_sub.mkdir(exist_ok=True)

        for wav in sorted(Path(folder).glob("*.wav")):
            feat = extract_features(wav)
            all_feats.append(feat)

            # Label nur für Training
            if name == "drone_real" or name == "drone_sim":
                all_labels.append(1)
            elif name == "no_drone":
                all_labels.append(0)

            np.save(out_sub / (wav.stem + ".npy"), feat)
            print(f"[OK] {wav.name}")

    all_feats = np.array(all_feats, dtype=np.float32)

    # GLOBAL NORMALIZATION
    mean = np.mean(all_feats, axis=0)
    scale = np.std(all_feats, axis=0)

    SCALER_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_JSON, "w") as fp:
        json.dump({"mean": mean.tolist(), "scale": scale.tolist()}, fp, indent=2)

    print("\n[OK] GLOBAL SCALER gespeichert:", SCALER_JSON)

    print("\n=== BUILD FEATURES V5 DONE ===")

if __name__ == "__main__":
    main()
