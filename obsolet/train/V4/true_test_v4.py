#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRUE-TEST V4 (Pfad‑B)
Kompatibel zu train_v4.py:
- Lädt Scaler (Z‑Norm)
- Normalisiert Features exakt wie im Training
- INT8 Inference
- Vollständige Statistik
"""

import numpy as np
import librosa
import json
from pathlib import Path
import tensorflow as tf

# ---------------------------------------------------------
# CONFIG (muss exakt zu train_v4 passen)
# ---------------------------------------------------------
SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

TFLITE_MODEL = Path("models/binary/model_binary_int8.tflite")
SCALER_JSON = Path("models/binary/scaler.json")
TRUE_DIR = Path("test/true_test")

# ---------------------------------------------------------
# Feature Extraction (40‑Dim Log‑Mel)
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
# Scaler laden (Z‑Norm)
# ---------------------------------------------------------
def load_scaler():
    with open(SCALER_JSON, "r") as fp:
        data = json.load(fp)

    mean = np.array(data["mean"], dtype=np.float32)
    scale = np.array(data["scale"], dtype=np.float32)

    print("[OK] Scaler geladen")
    return mean, scale

def apply_scaler(feat, mean, scale):
    return (feat - mean) / scale

# ---------------------------------------------------------
# INT8 Inference
# ---------------------------------------------------------
def infer_binary(interpreter, input_details, output_details, feat_norm):
    scale, zero_point = input_details[0]['quantization']

    qfeat = (feat_norm / scale + zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], qfeat.reshape(1, -1))

    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]['index'])[0][0]
    out_scale, out_zero = output_details[0]['quantization']

    score = (out - out_zero) * out_scale
    return float(score)

# ---------------------------------------------------------
# TRUE-TEST
# ---------------------------------------------------------
def run_true_test():
    # Modell laden
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[OK] INT8 Modell geladen")

    # Scaler laden
    mean, scale = load_scaler()

    print("\n=== TRUE-TEST START ===")

    scores = []
    labels = []
    files = sorted(TRUE_DIR.glob("*.wav"))

    for wav in files:
        feat = extract_features(wav)
        feat_norm = apply_scaler(feat, mean, scale)
        score = infer_binary(interpreter, input_details, output_details, feat_norm)

        label = "DRONE" if score >= 0.5 else "NO DRONE"
        scores.append(score)
        labels.append(label)

        print(f"{wav.name:40s}  Score={score:.3f}  → {label}")

    # Statistik
    n = len(scores)
    n_drone = sum(1 for l in labels if l == "DRONE")
    n_no = sum(1 for l in labels if l == "NO DRONE")

    false_negatives = n_no
    false_positive = 0

    avg_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    print("\n===================================")
    print("           TRUE-TEST Statistik")
    print("===================================")
    print(f"Anzahl Dateien:           {n}")
    print(f"DRONE erkannt:            {n_drone}")
    print(f"NO DRONE erkannt:         {n_no}")
    print(f"False Negatives:          {false_negatives}")
    print(f"False Positives:          {false_positive}")
    print(f"Score Durchschnitt:       {avg_score:.3f}")
    print(f"Score Minimum:            {min_score:.3f}")
    print(f"Score Maximum:            {max_score:.3f}")
    print("===================================")

    # Zero‑Input Test
    print("\n=== ZERO-INPUT TEST ===")
    zero = np.zeros(40, dtype=np.float32)
    zero_norm = apply_scaler(zero, mean, scale)
    score_zero = infer_binary(interpreter, input_details, output_details, zero_norm)
    print(f"Zero-Input Score={score_zero:.3f}")

    print("\n=== TRUE-TEST DONE ===")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    run_true_test()
