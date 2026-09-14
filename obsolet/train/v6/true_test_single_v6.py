#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRUE-TEST SINGLE V6
Testet genau EINE Drohne aus drone_real,
um versteckte Pipeline-Fehler auszuschließen.
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
# INT8 Inference
# ---------------------------------------------------------
def infer(interpreter, input_details, output_details, feat_norm):
    in_scale, in_zero = input_details[0]["quantization"]
    qfeat = (feat_norm / in_scale + in_zero).astype(np.int8)
    interpreter.set_tensor(input_details[0]["index"], qfeat.reshape(1, -1))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0][0]
    out_scale, out_zero = output_details[0]["quantization"]
    return float((out - out_zero) * out_scale)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("      TRUE-TEST SINGLE V6 START")
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

    # Feature extrahieren
    print(f"[INFO] Testdatei: {SINGLE_DRONE_FILE.name}")
    feat = extract_features(SINGLE_DRONE_FILE)
    feat_norm = (feat - mean) / scale

    # Inferenz
    score = infer(interpreter, input_details, output_details, feat_norm)
    label = "DRONE" if score >= 0.5 else "NO DRONE"

    print("\n=== Ergebnis ===")
    print(f"Score: {score:.3f}")
    print(f"Label: {label}")

    print("\n=== TRUE-TEST SINGLE V6 DONE ===")

if __name__ == "__main__":
    main()
