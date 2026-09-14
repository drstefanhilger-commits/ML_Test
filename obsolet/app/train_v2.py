#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic Training Pipeline V2 (Pfad‑B)
Binary classifier: DRONE (1) vs NO_DRONE (0)
Input: 40‑Dim Log‑Mel Features (from build_features.py)

- Voll deterministisch (Seeds gesetzt)
- Keine Hidden States
- Reproduzierbare Ergebnisse
- Echte INT8‑TFLite Vollquantisierung (INT8 Input/Output)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------------
# 1. Deterministische Seeds
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# 2. Feature‑Ordner
# ------------------------------------------------------------
FEATURE_DIRS = {
    "drone_real": "data/features/drone_real",
    "drone_sim": "data/features/drone_sim",
    "no_drone": "data/features/no_drone",
}

MODEL_DIR = "models/binary"
MODEL_H5 = os.path.join(MODEL_DIR, "model_binary.h5")
MODEL_TFLITE_INT8 = os.path.join(MODEL_DIR, "model_binary_int8.tflite")

# ------------------------------------------------------------
# 3. Laden der Features
# ------------------------------------------------------------
def load_features():
    X = []
    y = []

    # DRONE = 1
    for folder in ["drone_real", "drone_sim"]:
        fdir = FEATURE_DIRS[folder]
        files = sorted([f for f in os.listdir(fdir) if f.endswith(".npy")])
        for f in files:
            feat = np.load(os.path.join(fdir, f)).astype(np.float32)
            X.append(feat)
            y.append(1)

    # NO_DRONE = 0
    fdir = FEATURE_DIRS["no_drone"]
    files = sorted([f for f in os.listdir(fdir) if f.endswith(".npy")])
    for f in files:
        feat = np.load(os.path.join(fdir, f)).astype(np.float32)
        X.append(feat)
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[INFO] Loaded {len(X)} samples, feature_dim={X.shape[1]}")
    return X, y

# ------------------------------------------------------------
# 4. Modell definieren (Binary Classifier)
# ------------------------------------------------------------
def build_model(input_dim=40):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ------------------------------------------------------------
# 5. Repräsentatives Dataset für INT8‑Quantisierung
# ------------------------------------------------------------
def representative_dataset():
    # deterministisch, begrenzt, nur echte Feature‑Vektoren
    for folder in FEATURE_DIRS.values():
        files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        for f in files[:100]:
            feat = np.load(os.path.join(folder, f)).astype(np.float32)
            yield [feat.reshape(1, -1)]

# ------------------------------------------------------------
# 6. TFLite INT8 Export (echte Vollquantisierung)
# ------------------------------------------------------------
def export_tflite_int8(model):
    os.makedirs(MODEL_DIR, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Echte INT8‑Pfad: Input/Output INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(MODEL_TFLITE_INT8, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] Saved REAL INT8 TFLite model: {MODEL_TFLITE_INT8}")

# ------------------------------------------------------------
# 7. Training
# ------------------------------------------------------------
def train():
    print("======================================")
    print("       TRAINING PIPELINE V2")
    print("======================================")

    X, y = load_features()

    model = build_model(input_dim=X.shape[1])

    history = model.fit(
        X,
        y,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_H5)
    print(f"[OK] Saved Keras model: {MODEL_H5}")

    export_tflite_int8(model)

    print("======================================")
    print("       TRAINING V2 ERFOLGREICH")
    print("======================================")

# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    train()
