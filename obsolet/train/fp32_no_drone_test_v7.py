#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FP32 NO-DRONE TEST V7 — Pfad‑B, deterministisch

Testet das FP32‑Modell auf den No‑Drone‑Testdaten:

Input:
- test/false_test/*.wav

Output:
- Score pro Datei
- Durchschnittsscore
- Min/Max
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tensorflow as tf

# ---------------------------------------------------------
# Test‑Ordner (No‑Drone)
# ---------------------------------------------------------
TEST_DIR = Path("test/false_test")

# ---------------------------------------------------------
# Modell
# ---------------------------------------------------------
MODEL_PATH = Path("models/v7/model_binary_fp32_v7.h5")

# ---------------------------------------------------------
# DSP‑Parameter (identisch zu build_features_v7.py)
# ---------------------------------------------------------
SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000
FIXED_FRAMES = 256

# ---------------------------------------------------------
# Feature‑Extraktion (identisch zu FP32 FULL TEST V7)
# ---------------------------------------------------------
def extract_features(wav_path: Path):
    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        print(f"[ERR] {wav_path.name}: {e}")
        return None

    if sr != SR:
        print(f"[WARN] {wav_path.name}: SR={sr}, erwartet 48000")
        return None

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    frames = mel_db.shape[1]

    if frames < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    return mel_db.flatten().astype(np.float32)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("        FP32 NO-DRONE TEST V7")
    print("======================================")

    if not MODEL_PATH.exists():
        print(f"[ERR] Modell nicht gefunden: {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    wavs = sorted(TEST_DIR.glob("*.wav"))
    if not wavs:
        print(f"[ERR] Keine WAV-Dateien in {TEST_DIR}")
        return

    scores = []

    for wav in wavs:
        feat = extract_features(wav)
        if feat is None:
            continue

        pred = model.predict(feat.reshape(1, -1))[0][0]
        scores.append(pred)

        print(f"{wav.name:40s}  Score: {pred:.3f}")

    if scores:
        print("\n======================================")
        print(f"Anzahl Samples:     {len(scores)}")
        print(f"Score Durchschnitt: {np.mean(scores):.3f}")
        print(f"Score Minimum:      {np.min(scores):.3f}")
        print(f"Score Maximum:      {np.max(scores):.3f}")
        print("======================================")
    else:
        print("[ERR] Keine gültigen Features erzeugt.")

    print("======================================")
    print("        FP32 NO-DRONE TEST DONE")
    print("======================================")

if __name__ == "__main__":
    main()
