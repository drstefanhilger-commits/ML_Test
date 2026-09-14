#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRUE-TEST SINGLE FP32 V6
Testet eine echte Drohne im FP32-Modell,
um INT8-Quantisierungsfehler auszuschließen.
"""

import numpy as np
import librosa
import json
from pathlib import Path
import tensorflow as tf

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

SCALER_JSON = Path("models/binary/scaler_v5.json")
MODEL_FP32 = Path("models/binary/model_binary_v6.h5")

# Eine echte Drohne aus dem Training:
SINGLE_DRONE_FILE = Path("data/train_48k/drone_real/B_S2_D1_069-bebop_003_.wav")

# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Scaler
# ---------------------------------------------------------
def load_scaler():
    with open(SCALER_JSON, "r") as fp:
        d = json.load(fp)
    mean = np.array(d["mean"], dtype=np.float32)
    scale = np.array(d["scale"], dtype=np.float32)
    return mean, scale

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("      TRUE-TEST SINGLE FP32 V6 START")
    print("======================================")

    # FP32 Modell laden
    model = tf.keras.models.load_model(MODEL_FP32)
    print("[OK] FP32 Modell geladen")

    # Scaler laden
    mean, scale = load_scaler()
    print("[OK] Globaler Scaler geladen")

    # Feature extrahieren
    print(f"[INFO] Testdatei: {SINGLE_DRONE_FILE.name}")
    feat = extract_features(SINGLE_DRONE_FILE)
    feat_norm = (feat - mean) / scale

    # FP32 Inferenz
    score = float(model.predict(feat_norm.reshape(1, -1))[0][0])
    label = "DRONE" if score >= 0.5 else "NO DRONE"

    print("\n=== Ergebnis (FP32) ===")
    print(f"Score: {score:.3f}")
    print(f"Label: {label}")

    print("\n=== TRUE-TEST SINGLE FP32 V6 DONE ===")

if __name__ == "__main__":
    main()
