#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRAIN V7 — Pfad‑B, deterministisch

Trainiert ein Binary‑Classifier‑Modell auf den V7‑Features:

Ordner:
- data/features_v7/drone_real
- data/features_v7/no_drone

Erzeugt:
- models/v7/model_binary_fp32_v7.h5
- models/v7/model_binary_int8_v7.tflite
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# ---------------------------------------------------------
# Feature‑Ordner (V7)
# ---------------------------------------------------------
FEATURE_DIRS = {
    "drone_real": Path("data/features_v7/drone_real"),
    "no_drone":   Path("data/features_v7/no_drone"),
}

# ---------------------------------------------------------
# Output‑Ordner
# ---------------------------------------------------------
MODEL_DIR = Path("models/v7")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FP32_MODEL = MODEL_DIR / "model_binary_fp32_v7.h5"
INT8_MODEL = MODEL_DIR / "model_binary_int8_v7.tflite"

# ---------------------------------------------------------
# Determinismus
# ---------------------------------------------------------
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------
# Laden der Features
# ---------------------------------------------------------
def load_features():
    X = []
    y = []

    # Drone = 1
    for npy in sorted(FEATURE_DIRS["drone_real"].glob("*.npy")):
        feat = np.load(npy)
        X.append(feat)
        y.append(1)

    # No‑Drone = 0
    for npy in sorted(FEATURE_DIRS["no_drone"].glob("*.npy")):
        feat = np.load(npy)
        X.append(feat)
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[INFO] Features geladen: {X.shape[0]} Dateien")
    print(f"[INFO] Feature‑Shape: {X.shape}")

    return X, y

# ---------------------------------------------------------
# Modell definieren
# ---------------------------------------------------------
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ---------------------------------------------------------
# INT8‑Konvertierung
# ---------------------------------------------------------
def convert_to_int8(model, X):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative Dataset
    def rep_data():
        for i in range(min(200, len(X))):
            yield [X[i:i+1]]

    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(INT8_MODEL, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] INT8‑Modell gespeichert: {INT8_MODEL}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("              TRAIN V7")
    print("======================================")

    # Features laden
    X, y = load_features()

    # Modell bauen
    model = build_model(X.shape[1])

    # Training
    print("[INFO] Training startet…")
    history = model.fit(
        X, y,
        batch_size=32,
        epochs=20,
        validation_split=0.1,
        shuffle=True
    )

    # FP32‑Modell speichern
    model.save(FP32_MODEL)
    print(f"[INFO] FP32‑Modell gespeichert: {FP32_MODEL}")

    # INT8‑Konvertierung
    convert_to_int8(model, X)

    print("======================================")
    print("              TRAIN V7 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
