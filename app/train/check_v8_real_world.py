#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from pathlib import Path
import tensorflow as tf

from app.train.features_v8 import extract_features_v8

MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------------
# REAL-WORLD DIRECTORIES
# ---------------------------------------------------------
DRONE_REAL = Path("data/train_48k/drone_real")
NO_DRONE_REAL = Path("data/train/no_drone")

# ---------------------------------------------------------
# CLASSIFY SINGLE WAV
# ---------------------------------------------------------
def classify_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    feat = extract_features_v8(audio)
    score = float(model.predict(feat.reshape(1, -1))[0][0])
    return score

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=====================================")
    print("        CHECK V8 REAL-WORLD")
    print("=====================================")

    drone_scores = []
    nodrone_scores = []
    fn = 0
    fp = 0

    # -----------------------------
    # DRONE REAL
    # -----------------------------
    print("\n=== DRONE REAL ===")
    for p in sorted(DRONE_REAL.glob("*.wav")):
        score = classify_wav(p)
        drone_scores.append(score)
        label = "DRONE" if score > 0.5 else "NO-DRONE"
        if score <= 0.5:
            fn += 1
        print(f"{p.name:40s}  Score={score:.3f}  → {label}")

    # -----------------------------
    # NO-DRONE REAL
    # -----------------------------
    print("\n=== NO-DRONE REAL ===")
    for p in sorted(NO_DRONE_REAL.glob("*.wav")):
        score = classify_wav(p)
        nodrone_scores.append(score)
        label = "DRONE" if score > 0.5 else "NO-DRONE"
        if score > 0.5:
            fp += 1
        print(f"{p.name:40s}  Score={score:.3f}  → {label}")

    # -----------------------------
    # STATISTIK
    # -----------------------------
    print("\n=====================================")
    print("             STATISTIK V8")
    print("=====================================")

    print(f"DRONE REAL:     {len(drone_scores)} Dateien")
    print(f"NO-DRONE REAL:  {len(nodrone_scores)} Dateien")
    print("-------------------------------------")

    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print("-------------------------------------")

    if drone_scores:
        print(f"DRONE Score Min: {min(drone_scores):.3f}")
        print(f"DRONE Score Max: {max(drone_scores):.3f}")
        print(f"DRONE Score Avg: {np.mean(drone_scores):.3f}")

    if nodrone_scores:
        print(f"NO-DRONE Score Min: {min(nodrone_scores):.3f}")
        print(f"NO-DRONE Score Max: {max(nodrone_scores):.3f}")
        print(f"NO-DRONE Score Avg: {np.mean(nodrone_scores):.3f}")

    print("=====================================")


if __name__ == "__main__":
    main()
