#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRAIN V5 (GLOBAL NORMALIZATION + INT8 EXPORT)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

FEATURE_DIR = Path("data/features_v5")
SCALER_JSON = Path("models/binary/scaler_v5.json")

MODEL_DIR = Path("models/binary")
MODEL_H5 = MODEL_DIR / "model_binary_v5.h5"
MODEL_TFLITE = MODEL_DIR / "model_binary_v5_int8.tflite"

def load_scaler():
    with open(SCALER_JSON, "r") as fp:
        d = json.load(fp)
    return np.array(d["mean"]), np.array(d["scale"])

def load_features():
    X = []
    y = []

    for name in ["drone_real", "drone_sim", "no_drone"]:
        for f in sorted((FEATURE_DIR / name).glob("*.npy")):
            feat = np.load(f)
            X.append(feat)
            y.append(1 if name != "no_drone" else 0)

    return np.array(X), np.array(y)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(40,)),
        layers.Dense(64), layers.ReLU(),
        layers.Dense(32), layers.ReLU(),
        layers.Dense(1), layers.Activation("sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def export_int8(model, X_norm):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data():
        for i in range(min(150, len(X_norm))):
            # WICHTIG: float32 erzwingen
            x = X_norm[i].astype(np.float32).reshape(1, -1)
            yield [x]

    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] INT8 Modell gespeichert: {MODEL_TFLITE}")

def train():
    print("=== TRAIN V5 START ===")

    mean, scale = load_scaler()
    X, y = load_features()

    X_norm = (X - mean) / scale

    model = build_model()
    history = model.fit(X_norm, y, epochs=25, batch_size=32, validation_split=0.2)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_H5)
    print(f"[OK] H5 Modell gespeichert: {MODEL_H5}")

    export_int8(model, X_norm)

    print("=== TRAIN V5 DONE ===")

if __name__ == "__main__":
    train()
