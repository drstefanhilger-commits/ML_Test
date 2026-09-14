#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRUE-TEST V3 (Pfad‑B)
Kompatibel zu train_v3.py (INT8 Input/Output)
Erzeugt vollständige Statistik:
- Anzahl Dateien
- DRONE / NO DRONE
- False Negatives
- False Positives
- Score-Durchschnitt / Min / Max
- Zero-Input Test
"""

import numpy as np
import librosa
from pathlib import Path
import tensorflow as tf

# ---------------------------------------------------------
# CONFIG (muss exakt zu build_features.py passen)
# ---------------------------------------------------------
SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

TFLITE_MODEL = Path("models/binary/model_binary_int8.tflite")
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
# INT8 Quantisierung
# ---------------------------------------------------------
def quantize(feat, scale, zero_point):
    return (feat / scale + zero_point).astype(np.int8)

# ---------------------------------------------------------
# Load TFLite Model
# ---------------------------------------------------------
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[OK] INT8 TFLite Modell geladen")
    print("Input dtype:", input_details[0]["dtype"])
    print("Output dtype:", output_details[0]["dtype"])
    print("Quantization:", input_details[0]["quantization"])

    return interpreter, input_details, output_details

# ---------------------------------------------------------
# Inference (INT8)
# ---------------------------------------------------------
def infer_binary(interpreter, input_details, output_details, feat):
    scale, zero_point = input_details[0]['quantization']

    qfeat = quantize(feat, scale, zero_point)
    interpreter.set_tensor(input_details[0]['index'], qfeat.reshape(1, -1))

    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]['index'])[0][0]

    out_scale, out_zero = output_details[0]['quantization']
    score = (out - out_zero) * out_scale

    return float(score)

# ---------------------------------------------------------
# Main TRUE-Test + Statistik
# ---------------------------------------------------------
def run_true_test():
    interpreter, input_details, output_details = load_interpreter()

    print("\n=== TRUE-TEST START ===")

    scores = []
    labels = []
    files = sorted(TRUE_DIR.glob("*.wav"))

    for wav in files:
        feat = extract_features(wav)
        score = infer_binary(interpreter, input_details, output_details, feat)

        label = "DRONE" if score >= 0.5 else "NO DRONE"
        scores.append(score)
        labels.append(label)

        print(f"{wav.name:40s}  Score={score:.3f}  → {label}")

    # -----------------------------------------------------
    # Statistik
    # -----------------------------------------------------
    n = len(scores)
    n_drone = sum(1 for l in labels if l == "DRONE")
    n_no = sum(1 for l in labels if l == "NO DRONE")

    # TRUE-TEST enthält nur echte Drohnen → NO DRONE = False Negative
    false_negatives = n_no
    false_positive = 0  # TRUE-TEST enthält keine NO-DRONE Dateien

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

    # -----------------------------------------------------
    # Zero-Input Test
    # -----------------------------------------------------
    print("\n=== ZERO-INPUT TEST ===")
    zero = np.zeros(40, dtype=np.float32)
    score_zero = infer_binary(interpreter, input_details, output_details, zero)
    print(f"Zero-Input Score={score_zero:.3f}")

    print("\n=== TRUE-TEST DONE ===")

    print("Input dtype:", input_details[0]["dtype"])
    print("Output dtype:", output_details[0]["dtype"])
    print("Quantization:", input_details[0]["quantization"])


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    run_true_test()
