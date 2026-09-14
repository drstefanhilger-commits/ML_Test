#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRUE-TEST V5 (GLOBAL NORMALIZATION)
"""

import numpy as np
import librosa
import json
from pathlib import Path
import tensorflow as tf

SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

SCALER_JSON = Path("models/binary/scaler_v5.json")
TFLITE_MODEL = Path("models/binary/model_binary_v5_int8.tflite")
TRUE_DIR = Path("test/true_test")

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

def load_scaler():
    with open(SCALER_JSON, "r") as fp:
        d = json.load(fp)
    return np.array(d["mean"]), np.array(d["scale"])

def infer(interpreter, input_details, output_details, feat_norm):
    scale, zero = input_details[0]["quantization"]
    qfeat = (feat_norm / scale + zero).astype(np.int8)
    interpreter.set_tensor(input_details[0]["index"], qfeat.reshape(1, -1))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0][0]
    out_scale, out_zero = output_details[0]["quantization"]
    return float((out - out_zero) * out_scale)

def main():
    mean, scale = load_scaler()

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scores = []
    labels = []

    for wav in sorted(TRUE_DIR.glob("*.wav")):
        feat = extract_features(wav)
        feat_norm = (feat - mean) / scale
        score = infer(interpreter, input_details, output_details, feat_norm)
        label = "DRONE" if score >= 0.5 else "NO DRONE"
        scores.append(score)
        labels.append(label)
        print(f"{wav.name:40s} Score={score:.3f} → {label}")

    print("\n=== TRUE-TEST V5 DONE ===")

if __name__ == "__main__":
    main()
