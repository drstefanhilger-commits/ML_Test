#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic Training Pipeline V4 (Pfad‑B)
Mit:
- Z‑Norm Feature‑Normalisierung (StandardScaler)
- Darstellung der Trainings‑Ergebnisse
- Speicherung des Scalers
- INT8‑TFLite Export
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

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
SCALER_JSON = os.path.join(MODEL_DIR, "scaler.json")

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
# 4. INT8‑kompatibles Modell
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 5. Repräsentatives Dataset für INT8‑Quantisierung
# ------------------------------------------------------------
def representative_dataset(X_norm):
    for i in range(min(150, len(X_norm))):
        yield [X_norm[i].reshape(1, -1)]

# ------------------------------------------------------------
# 6. TFLite INT8 Export
# ------------------------------------------------------------
def export_tflite_int8(model, X_norm):
    os.makedirs(MODEL_DIR, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(X_norm)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(MODEL_TFLITE_INT8, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] Saved REAL INT8 TFLite model: {MODEL_TFLITE_INT8}")

# ------------------------------------------------------------
# 7. Darstellung der Trainings‑Ergebnisse
# ------------------------------------------------------------
def print_training_stats(X, X_norm, y, history, scaler):
    print("\n======================================")
    print("        TRAINING STATISTIK V4")
    print("======================================")

    print(f"Samples total: {len(X)}")
    print(f"DRONE:        {np.sum(y==1)}")
    print(f"NO_DRONE:     {np.sum(y==0)}")

    print("\nFeature Statistik (vor Normierung):")
    print(f"  Mean: {np.mean(X):.3f}")
    print(f"  Std:  {np.std(X):.3f}")
    print(f"  Min:  {np.min(X):.3f}")
    print(f"  Max:  {np.max(X):.3f}")

    print("\nFeature Statistik (nach Normierung):")
    print(f"  Mean: {np.mean(X_norm):.3f}")
    print(f"  Std:  {np.std(X_norm):.3f}")
    print(f"  Min:  {np.min(X_norm):.3f}")
    print(f"  Max:  {np.max(X_norm):.3f}")

    print("\nTraining Loss/Accuracy:")
    print(f"  Final Loss:     {history.history['loss'][-1]:.4f}")
    print(f"  Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Val Loss:       {history.history['val_loss'][-1]:.4f}")
    print(f"  Val Accuracy:   {history.history['val_accuracy'][-1]:.4f}")

    print("\nScaler Parameter:")
    print(f"  Mean: {scaler.mean_.tolist()}")
    print(f"  Scale: {scaler.scale_.tolist()}")

    print("======================================")

# ------------------------------------------------------------
# 8. Training
# ------------------------------------------------------------
def train():
    print("======================================")
    print("       TRAINING PIPELINE V4")
    print("======================================")

    X, y = load_features()

    # ------------------------------
    # Z‑Norm Normalisierung
    # ------------------------------
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Scaler speichern
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_JSON, "w") as fp:
        json.dump({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }, fp, indent=2)

    print("[OK] Scaler gespeichert:", SCALER_JSON)

    # ------------------------------
    # Modell bauen
    # ------------------------------
    model = build_model(input_dim=X.shape[1])

    # ------------------------------
    # Training
    # ------------------------------
    history = model.fit(
        X_norm,
        y,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
    )

    model.save(MODEL_H5)
    print(f"[OK] Saved Keras model: {MODEL_H5}")

    # ------------------------------
    # INT8 Export
    # ------------------------------
    export_tflite_int8(model, X_norm)

    # ------------------------------
    # Darstellung
    # ------------------------------
    print_training_stats(X, X_norm, y, history, scaler)

    print("======================================")
    print("       TRAINING V4 ERFOLGREICH")
    print("======================================")

# ------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    train()
