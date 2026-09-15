#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import tensorflow as tf

from app.sds.sds_drone_generator import make_sds_drone_frame
from app.train.features_v8 import extract_features_v8

MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
model = tf.keras.models.load_model(MODEL_PATH)


def classify_sds_v8(angle, distance, noise):
    mic = make_sds_drone_frame(angle, distance, noise)
    audio = np.mean(mic, axis=0)
    feat = extract_features_v8(audio)
    score = float(model.predict(feat.reshape(1, -1))[0][0])
    return score


def run_scenario(name, angle, distance, noise, n=50):
    scores = []
    print(f"\n=== {name} | angle={angle}° dist={distance}m noise={noise:.2f} ===")
    for i in range(n):
        s = classify_sds_v8(angle, distance, noise)
        scores.append(s)
        print(f"Run {i+1:3d}: Score={s:.3f}")

    scores = np.array(scores)
    print(f"--- {name} STATS ---")
    print(f"Mean: {scores.mean():.3f}")
    print(f"Min : {scores.min():.3f}")
    print(f"Max : {scores.max():.3f}")
    print(f"Std : {scores.std():.3f}")


def main():
    run_scenario("A_far_heavy_noise", angle=135, distance=200, noise=0.20, n=50)
    run_scenario("B_mid_noise",       angle=90,  distance=150, noise=0.10, n=50)
    run_scenario("C_mid_low_noise",   angle=45,  distance=100, noise=0.05, n=50)
    run_scenario("D_close_clean",     angle=0,   distance=50,  noise=0.00, n=50)


if __name__ == "__main__":
    main()
