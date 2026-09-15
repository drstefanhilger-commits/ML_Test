#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from pathlib import Path
import tensorflow as tf

from app.train.features_v8 import extract_features_v8

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------------
# V9 NO-DRONE SIMULATION TEST DIRECTORY
# ---------------------------------------------------------
NO_DRONE_SIM_V9 = Path("data/test_sim_v9/no_drone_sim_safe")

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
    print("     CHECK V8 ON V9 NO-DRONE SIM")
    print("=====================================")

    nodrone_scores = []
    fp = 0

    # -----------------------------------------------------
    # NO-DRONE SIM V9
    # -----------------------------------------------------
    print("\n=== NO-DRONE SIM V9 ===")
    for p in sorted(NO_DRONE_SIM_V9.glob("*.wav")):
        score = classify_wav(p)
        nodrone_scores.append(score)

        label = "DRONE" if score > 0.5 else "NO-DRONE"

        if score > 0.5:
            fp += 1
            print(f"[FP] {p.name:40s} Score={score:.3f}")
        else:
            print(f"{p.name:40s} Score={score:.3f} ({label})")

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------
    print("\n=====================================")
    print("             STATISTICS")
    print("=====================================")

    print(f"No-Drone Sim Files:  {len(nodrone_scores)}")
    print(f"False Positives:     {fp}")
    print("-------------------------------------")

    if nodrone_scores:
        print(f"No-Drone Avg:        {np.mean(nodrone_scores):.3f}")
        print(f"No-Drone Min:        {np.min(nodrone_scores):.3f}")
        print(f"No-Drone Max:        {np.max(nodrone_scores):.3f}")

    print("=====================================")


if __name__ == "__main__":
    main()
