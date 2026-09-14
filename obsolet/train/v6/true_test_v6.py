#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRUE-TEST V6 (GLOBAL NORMALIZATION, ROBUST)
- Nutzt globalen Scaler aus scaler_v5.json
- Nutzt INT8 Modell model_binary_v6_int8.tflite
- Feature-Extraktion identisch zu build_features_v5/train_v6
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
TFLITE_MODEL = Path("models/binary/model_binary_v6_int8.tflite")
TRUE_DIR = Path("test/true_test")

# ---------------------------------------------------------
# Feature Extraction (identisch zu v5/v6)
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

def apply_scaler(feat, mean, scale):
    return (feat - mean) / scale

# ---------------------------------------------------------
# INT8 Inference
# ---------------------------------------------------------
def infer(interpreter, input_details, output_details, feat_norm):
    in_scale, in_zero = input_details[0]["quantization"]

    qfeat = (feat_norm / in_scale + in_zero).astype(np.int8)
    interpreter.set_tensor(input_details[0]["index"], qfeat.reshape(1, -1))

    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])[0][0]
    out_scale, out_zero = output_details[0]["quantization"]

    score = (out - out_zero) * out_scale
    return float(score)

# ---------------------------------------------------------
# TRUE-TEST V6
# ---------------------------------------------------------
def main():
    print("======================================")
    print("           TRUE-TEST V6 START")
    print("======================================")

    # Modell laden
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[OK] INT8 Modell geladen")

    # Scaler laden
    mean, scale = load_scaler()
    print("[OK] Globaler Scaler geladen")

    scores = []
    labels = []

    files = sorted(TRUE_DIR.glob("*.wav"))
    print(f"[INFO] TRUE-TEST Dateien: {len(files)}")

    for wav in files:
        feat = extract_features(wav)
        feat_norm = apply_scaler(feat, mean, scale)
        score = infer(interpreter, input_details, output_details, feat_norm)

        label = "DRONE" if score >= 0.5 else "NO DRONE"
        scores.append(score)
        labels.append(label)

        print(f"{wav.name:40s} Score={score:.3f} → {label}")

    # Statistik
    n = len(scores)
    n_drone = sum(1 for l in labels if l == "DRONE")
    n_no = sum(1 for l in labels if l == "NO DRONE")

    print("\n======================================")
    print("           TRUE-TEST V6 Statistik")
    print("======================================")
    print(f"Anzahl Dateien:           {n}")
    print(f"DRONE erkannt:            {n_drone}")
    print(f"NO DRONE erkannt:         {n_no}")
    print(f"Score Durchschnitt:       {np.mean(scores):.3f}")
    print(f"Score Minimum:            {np.min(scores):.3f}")
    print(f"Score Maximum:            {np.max(scores):.3f}")
    print("======================================")

    # Zero-Input Test
    print("\n=== ZERO-INPUT TEST ===")
    zero = np.zeros(40, dtype=np.float32)
    zero_norm = apply_scaler(zero, mean, scale)
    score_zero = infer(interpreter, input_details, output_details, zero_norm)
    print(f"Zero-Input Score={score_zero:.3f}")

    print("\n=== TRUE-TEST V6 DONE ===")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
