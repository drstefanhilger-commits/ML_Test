#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from pathlib import Path

from sds_drone_generator import make_sds_drone_frame
from app.features.features_v8 import extract_features_v8

MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
model = tf.keras.models.load_model(MODEL_PATH)

angles = [0, 45, 90, 135, 180, 225, 270, 315]
distances = [20, 30, 50, 75, 100, 150, 200]
noise_levels = [0.00, 0.05, 0.10, 0.20]

scores = []
false_negatives = 0
total = 0


def classify_sds_v8(angle, distance, noise):
    mic = make_sds_drone_frame(angle, distance, noise)
    audio = np.mean(mic, axis=0)
    feat = extract_features_v8(audio)
    score = float(model.predict(feat.reshape(1, -1))[0][0])
    return score


print("====================================================")
print("      SDS FAR-RANGE Angle–Distance–Noise Test V8")
print("====================================================")

for a in angles:
    print(f"\n=== Angle {a}° ===")
    for d in distances:
        for n in noise_levels:
            score = classify_sds_v8(a, d, n)
            label = "DRONE" if score > 0.5 else "NO DRONE"

            print(f"Dist={d:3.0f}m  Noise={n:.2f}  → Score={score:.3f} ({label})")

            scores.append(score)
            total += 1
            if score <= 0.5:
                false_negatives += 1


scores_np = np.array(scores)
avg_score = float(np.mean(scores_np))
min_score = float(np.min(scores_np))
max_score = float(np.max(scores_np))

fn_rate = false_negatives / total
drone_count = sum(1 for s in scores if s > 0.5)
nodrone_count = sum(1 for s in scores if s <= 0.5)
drone_rate = drone_count / total
nodrone_rate = nodrone_count / total

print("\n====================================================")
print("                 STATISTIK V8")
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
