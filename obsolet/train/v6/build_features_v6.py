#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD FEATURES V6
- Extrahiert Features aus TRAIN + TEST Ordnern
- TRAIN: drone_real_train + no_drone
- TEST:  true_test + false_test
- Globale Normierung wird NUR aus TRAIN berechnet
"""

import numpy as np
import librosa
import json
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

DATA_DIRS = {
    "drone_real_train": "data/train_48k/drone_real_train",
    "no_drone":         "data/train_48k/no_drone",
    "true_test":        "test/true_test",
    "false_test":       "test/false_test"
}

OUT_DIR = Path("data/features_v6")
OUT_DIR.mkdir(exist_ok=True)

SCALER_JSON = Path("models/binary/scaler_v6.json")

# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

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

    logmel = librosa.power_to_db(mel, ref=np.max)
    feat = np.mean(logmel, axis=1)

    return feat.astype(np.float32)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("         BUILD FEATURES V6 START")
    print("======================================")

    all_train_feats = []

    # -----------------------------------------------------
    # Feature-Erzeugung
    # -----------------------------------------------------
    for name, folder in DATA_DIRS.items():
        print(f"\n--- {name} ---")

        out_sub = OUT_DIR / name
        out_sub.mkdir(exist_ok=True)

        wav_files = sorted(Path(folder).glob("*.wav"))
        if not wav_files:
            print(f"[WARN] Keine WAV-Dateien gefunden in {folder}")
            continue

        for wav in wav_files:
            feat = extract_features(wav)
            np.save(out_sub / (wav.stem + ".npy"), feat)
            print(f"[OK] {wav.name}")

            # TRAIN-Daten bestimmen den Scaler
            if name in ["drone_real_train", "no_drone"]:
                all_train_feats.append(feat)

    # -----------------------------------------------------
    # Globale Normierung aus TRAIN-Daten
    # -----------------------------------------------------
    if len(all_train_feats) == 0:
        print("[ERR] Keine TRAIN-Features gefunden! Split prüfen!")
        return

    all_train_feats = np.array(all_train_feats, dtype=np.float32)

    mean = np.mean(all_train_feats, axis=0)
    scale = np.std(all_train_feats, axis=0)

    SCALER_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_JSON, "w") as fp:
        json.dump({"mean": mean.tolist(), "scale": scale.tolist()}, fp, indent=2)

    print("\n[OK] Globaler Scaler gespeichert:", SCALER_JSON)
    print("======================================")
    print("         BUILD FEATURES V6 DONE")
    print("======================================")

# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
