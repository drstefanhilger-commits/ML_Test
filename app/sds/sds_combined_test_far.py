#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

from sds_drone_generator import make_sds_drone_frame

MODEL_PATH = Path("models/v7/model_binary_fp32_v7.h5")

SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FIXED_FRAMES = 256

def extract_features(audio):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=20,
        fmax=20000,
        power=2.0
    )

    # dB ohne Normalisierung auf max → Distanz bleibt sichtbar
    mel_db = librosa.power_to_db(mel, ref=1.0)

    # Padding / Cutting wie vorher
    frames = mel_db.shape[1]
    if frames < FIXED_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0,FIXED_FRAMES-frames)), mode="constant")
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    # Flatten → exakt 10240 Werte
    feat = mel_db.flatten().astype(np.float32)

    # RMS separat berechnen
    rms = np.sqrt(np.mean(audio**2))

    return feat, rms


def classify_sds(angle, distance, noise):
    mic = make_sds_drone_frame(angle, distance, noise)

    # Kanal-Mix wie vorher
    audio = np.mean(mic, axis=0)

    # neue Feature-Pipeline
    feat, rms = extract_features(audio)

    # Modell erwartet zwei Inputs:
    # 1) Feature-Vektor (10240)
    # 2) RMS (1 Wert)
    score = float(model.predict([feat.reshape(1,-1), np.array([[rms]])])[0][0])

    return score


angles = [0, 45, 90, 135, 180, 225, 270, 315]
distances = [50, 75, 100, 150, 200]
noise_levels = [0.0, 0.1, 0.2]

scores = []
false_negatives = 0
total = 0

print("====================================================")
print("      SDS FAR-RANGE Angle–Distance–Noise Test")
print("====================================================")

for a in angles:
    print(f"\n=== Angle {a}° ===")
    for d in distances:
        for n in noise_levels:
            score = classify_sds(a, d, n)
            label = "DRONE" if score > 0.5 else "NO DRONE"

            print(f"Dist={d:3.0f}m  Noise={n:.2f}  → Score={score:.3f} ({label})")

            scores.append(score)
            total += 1
            if score <= 0.5:
                false_negatives += 1

# Statistik
scores_np = np.array(scores)
avg_score = float(np.mean(scores_np))
min_score = float(np.min(scores_np))
max_score = float(np.max(scores_np))

# False-Negative-Rate
fn_rate = false_negatives / total

# Drone / No-Drone Statistik
drone_count = sum(1 for s in scores if s > 0.5)
nodrone_count = sum(1 for s in scores if s <= 0.5)

drone_rate = drone_count / total
nodrone_rate = nodrone_count / total

print("\n====================================================")
print("                 STATISTIK")
print("====================================================")
print(f"Anzahl Tests:            {total}")
print(f"False Negatives:         {false_negatives}")
print(f"False-Negative-Rate:     {fn_rate:.3f}")
print(f"Score Durchschnitt:      {avg_score:.3f}")
print(f"Score Minimum:           {min_score:.3f}")
print(f"Score Maximum:           {max_score:.3f}")
print("----------------------------------------------------")
print(f"DRONE erkannt:           {drone_count}")
print(f"NO-DRONE erkannt:        {nodrone_count}")
print(f"DRONE-Rate:              {drone_rate:.3f}")
print(f"NO-DRONE-Rate:           {nodrone_rate:.3f}")
print("====================================================")
