#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FP32 NO-DRONE TEST V6
Testet ALLE No-Drone WAVs, die NICHT im Training waren.
Quelle: data/train_48k/no_drone_test/
- FP32 Modell: model_binary_v6.h5
- Scaler: scaler_v6.json
- Ausgabe: Score, Label, Statistik
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

TEST_DIR = Path("data/train_48k/no_drone_test")

MODEL_FP32 = Path("models/binary/model_binary_v6.h5")
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
# Scaler laden
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
    print("        FP32 NO-DRONE TEST V6 START")
    print("======================================")

    # Modell laden
    model = tf.keras.models.load_model(MODEL_FP32)
    print("[OK] FP32 Modell geladen")

    # Scaler laden
    mean, scale = load_scaler()
    print("[OK] Scaler geladen")

    # Testdateien
    test_files = sorted(TEST_DIR.glob("*.wav"))
    if not test_files:
        print("[ERR] Keine No-Drone Testdateien gefunden:", TEST_DIR)
        return

    print(f"[INFO] Testdateien: {len(test_files)}")

    results = []
    scores = []
    correct = 0
    wrong = []

    for wav in test_files:
        print(f"\n[INFO] Teste: {wav.name}")

        feat = extract_features(wav)
        feat_norm = (feat - mean) / scale

        score = float(model.predict(feat_norm.reshape(1, -1))[0][0])
        label = "DRONE" if score >= 0.5 else "NO DRONE"

        print(f"Score: {score:.3f}")
        print(f"Label: {label}")

        results.append((wav.name, score, label))
        scores.append(score)

        if score < 0.5:
            correct += 1
        else:
            wrong.append((wav.name, score))

    # -----------------------------------------------------
    # Statistik
    # -----------------------------------------------------
    total = len(test_files)
    false_positives = len(wrong)
    accuracy = correct / total

    print("\n======================================")
    print("        FP32 NO-DRONE TEST V6 SUMMARY")
    print("======================================")

    for name, score, label in results:
        print(f"{name:40s}  Score={score:.3f}  Label={label}")

    print("\n======================================")
    print("              STATISTIK")
    print("======================================")
    print(f"Anzahl No-Drone Samples:     {total}")
    print(f"Korrekt erkannt (NO DRONE):  {correct}")
    print(f"Falsch erkannt (DRONE):      {false_positives}")
    print(f"False-Positive-Rate:         {false_positives/total:.3f}")
    print(f"Score Minimum:               {min(scores):.3f}")
    print(f"Score Maximum:               {max(scores):.3f}")
    print(f"Score Durchschnitt:          {np.mean(scores):.3f}")

    if wrong:
        print("\nFalsch erkannte No-Drone Samples:")
        for name, score in wrong:
            print(f"  {name:40s} Score={score:.3f}")
    else:
        print("\nAlle No-Drone Samples wurden korrekt erkannt!")

    print("======================================")
    print("        FP32 NO-DRONE TEST V6 DONE")
    print("======================================")

# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
