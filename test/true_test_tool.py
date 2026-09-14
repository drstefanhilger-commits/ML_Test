#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

MODEL_PATH = Path("models/binary/model_binary_int8.tflite")
TRUE_TEST_DIR = Path("data/train/true_test")


# ---------------------------------------------------------
# Feature Extraction (STM32-kompatibel)
# ---------------------------------------------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < FRAME_LEN:
        y = np.pad(y, (0, FRAME_LEN - len(y)))
    else:
        y = y[:FRAME_LEN]

    window = np.hanning(FRAME_LEN)
    S = librosa.stft(
        y,
        n_fft=FRAME_LEN,
        hop_length=FRAME_LEN // 2,
        window=window,
        center=False
    )

    mag = np.abs(S)[:129, :]
    mel = MEL_FILTERBANK @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------
# Load INT8 TFLite Model
# ---------------------------------------------------------
def load_model():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------
# INT8 Prediction
# ---------------------------------------------------------
def run_model(interpreter, x_float):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantisierung
    scale, zero = input_details[0]['quantization']
    x_q = x_float / scale + zero
    x_q = np.clip(x_q, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], [x_q])
    interpreter.invoke()

    out_q = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Dequantisierung
    scale_o, zero_o = output_details[0]['quantization']
    score = (out_q - zero_o) * scale_o

    return float(score)


# ---------------------------------------------------------
# Main TRUE-TEST
# ---------------------------------------------------------
def run_true_test():
    interpreter = load_model()

    wavs = sorted(TRUE_TEST_DIR.glob("*.wav"))
    print(f"Gefundene True-Test WAV-Dateien: {len(wavs)}")
    print("Starte True-Test Analyse...\n")

    scores = []
    false_negatives = 0

    for wav in wavs:
        feat = extract_features(wav)
        score = run_model(interpreter, feat)

        scores.append(score)

        label = "DRONE" if score >= 0.5 else "NO DRONE"
        if score < 0.5:
            false_negatives += 1

        print(f"=== TRUE-SOUND: {wav.name} ===")
        print(f"True-Test -> {score:.3f}  ({label})\n")

    # ---------------------------------------------------------
    # Statistik
    # ---------------------------------------------------------
    scores = np.array(scores)
    total = len(scores)
    fn_rate = false_negatives / total if total > 0 else 0

    print("\n====================================")
    print("           TRUE-TEST Statistik       ")
    print("====================================")
    print(f"Anzahl Dateien:           {total}")
    print(f"False Negatives:          {false_negatives}")
    print(f"False-Negative-Rate:      {fn_rate:.3f}")
    print(f"Score Durchschnitt:       {scores.mean():.3f}")
    print(f"Score Minimum:            {scores.min():.3f}")
    print(f"Score Maximum:            {scores.max():.3f}")
    print("====================================\n")


if __name__ == "__main__":
    run_true_test()
