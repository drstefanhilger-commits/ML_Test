#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from pathlib import Path
import tensorflow as tf

from app.train.features_v8 import extract_features_v8

MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
model = tf.keras.models.load_model(MODEL_PATH)

DRONE_SIM_V9 = Path("data/test_sim_v9/drone_sim")
NO_DRONE_SIM_V9 = Path("data/test_sim_v9/no_drone_sim")

def classify_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    feat = extract_features_v8(audio)
    score = float(model.predict(feat.reshape(1, -1))[0][0])
    return score

def main():
    print("=====================================")
    print("        CHECK V8 ON SIMULATION V9")
    print("=====================================")

    drone_scores = []
    nodrone_scores = []
    fn = 0
    fp = 0

    print("\n=== DRONE SIM V9 ===")
    for p in sorted(DRONE_SIM_V9.glob("*.wav")):
        score = classify_wav(p)
        drone_scores.append(score)
        if score <= 0.5:
            fn += 1
        print(f"{p.name:40s}  Score={score:.3f}")

    print("\n=== NO-DRONE SIM V9 ===")
    for p in sorted(NO_DRONE_SIM_V9.glob("*.wav")):
        score = classify_wav(p)
        nodrone_scores.append(score)
        if score > 0.5:
            fp += 1
        print(f"{p.name:40s}  Score={score:.3f}")

    print("\n=====================================")
    print("             STATISTICS")
    print("=====================================")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print("-------------------------------------")
    print(f"Drone Avg: {np.mean(drone_scores):.3f}")
    print(f"Drone Min: {np.min(drone_scores):.3f}")
    print(f"Drone Max: {np.max(drone_scores):.3f}")
    print("-------------------------------------")
    print(f"No-Drone Avg: {np.mean(nodrone_scores):.3f}")
    print(f"No-Drone Min: {np.min(nodrone_scores):.3f}")
    print(f"No-Drone Max: {np.max(nodrone_scores):.3f}")
    print("=====================================")

if __name__ == "__main__":
    main()
