#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRAIN V6 (GLOBAL NORMALIZATION, TRAIN/TEST SPLIT)
- Drohne: drone_real_train
- Kein drone_real_test im Training!
- Kein true_test im Training!
- Globale Z-Norm aus scaler_v6.json
- INT8-TFLite Export
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

FEATURE_DIR = Path("data/features_v6")
SCALER_JSON = Path("models/binary/scaler_v6.json")

MODEL_DIR = Path("models/binary")
MODEL_H5 = MODEL_DIR / "model_binary_v6.h5"
MODEL_TFLITE = MODEL_DIR / "model_binary_v6_int8.tflite"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------
# Scaler laden
# ---------------------------------------------------------
def load_scaler():
    with open(SCALER_JSON, "r") as fp:
        d = json.load(fp)
    mean = np.array(d["mean"], dtype=np.float32)
    scale = np.array(d["scale"], dtype=np.float32)
    return mean, scale

# ---------------------------------------------------------
# Features laden (nur TRAIN!)
# ---------------------------------------------------------
def load_features_v6():
    X = []
    y = []

    # Drohne: NUR drone_real_train
    for f in sorted((FEATURE_DIR / "drone_real_train").glob("*.npy")):
        feat = np.load(f).astype(np.float32)
        X.append(feat)
        y.append(1)

    # No-Drone
    for f in sorted((FEATURE_DIR / "no_drone").glob("*.npy")):
        feat = np.load(f).astype(np.float32)
        X.append(feat)
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[INFO] TRAIN V6: Loaded {len(X)} samples, feature_dim={X.shape[1]}")
    print(f"[INFO] DRONE samples: {np.sum(y==1)}")
    print(f"[INFO] NO_DRONE samples: {np.sum(y==0)}")

    return X, y

# ---------------------------------------------------------
# Modell
# ---------------------------------------------------------
def build_model(input_dim=40):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64),
        layers.ReLU(),
        layers.Dense(32),
        layers.ReLU(),
        layers.Dense(1),
        layers.Activation("sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ---------------------------------------------------------
# INT8 Export
# ---------------------------------------------------------
def export_int8(model, X_norm):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data():
        for i in range(min(150, len(X_norm))):
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

# ---------------------------------------------------------
# Training V6
# ---------------------------------------------------------
def train_v6():
    print("======================================")
    print("           TRAINING V6 START")
    print("======================================")

    mean, scale = load_scaler()
    X, y = load_features_v6()

    # Globale Normierung
    X_norm = (X - mean) / scale

    model = build_model(input_dim=X.shape[1])

    history = model.fit(
        X_norm,
        y,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_H5)
    print(f"[OK] H5 Modell gespeichert: {MODEL_H5}")

    export_int8(model, X_norm)

    print("\n======================================")
    print("           TRAINING V6 DONE")
    print("======================================")
    print(f"Final Loss:     {history.history['loss'][-1]:.4f}")
    print(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Val Loss:       {history.history['val_loss'][-1]:.4f}")
    print(f"Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    train_v6()
