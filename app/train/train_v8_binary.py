#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import soundfile as sf
import random

from app.train.features_v8 import extract_features_v8

# ---------------------------------------------------------
# MODEL OUTPUT PATH
# ---------------------------------------------------------
MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
INPUT_DIM = 10243

# ---------------------------------------------------------
# TRAINING DATA DIRECTORIES
# ---------------------------------------------------------
BASE = Path("data/train_v8")

DRONE_REAL = Path("data/train_48k/drone_real")     # deine echten Drohnen
NO_DRONE_REAL = Path("data/train/no_drone")        # dog, fan_noise, helicopter, wind

DRONE_SIM = BASE / "drone_sim"
NO_DRONE_SIM = BASE / "no_drone_sim"


# ---------------------------------------------------------
# LOAD ALL WAV PATHS + LABELS
# ---------------------------------------------------------
def collect_wavs():
    wav_paths = []
    labels = []

    # -----------------------------
    # DRONE_REAL (label = 1)
    # -----------------------------
    if DRONE_REAL.exists():
        for p in DRONE_REAL.glob("*.wav"):
            wav_paths.append(p)
            labels.append(1)

    # -----------------------------
    # DRONE_SIM (label = 1)
    # -----------------------------
    for p in DRONE_SIM.glob("*.wav"):
        wav_paths.append(p)
        labels.append(1)

    # -----------------------------
    # NO_DRONE_REAL (label = 0)
    # -----------------------------
    if NO_DRONE_REAL.exists():
        for p in NO_DRONE_REAL.glob("*.wav"):
            wav_paths.append(p)
            labels.append(0)

    # -----------------------------
    # NO_DRONE_SIM (label = 0)
    # -----------------------------
    for p in NO_DRONE_SIM.glob("*.wav"):
        wav_paths.append(p)
        labels.append(0)

    # Shuffle for randomness
    combined = list(zip(wav_paths, labels))
    random.shuffle(combined)
    wav_paths, labels = zip(*combined)

    return list(wav_paths), list(labels)


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def load_dataset(wav_paths, labels):
    X, y = [], []
    for p, lab in zip(wav_paths, labels):
        audio, sr = sf.read(p)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        feat = extract_features_v8(audio)
        X.append(feat)
        y.append(lab)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------
def build_model_v8():
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_DIM,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def main():
    print("=====================================")
    print("        TRAINING PIPELINE V8")
    print("=====================================")

    wav_paths, labels = collect_wavs()
    print(f"Gefundene WAVs: {len(wav_paths)}")
    print(f"Drohnen:        {sum(labels)}")
    print(f"No-Drone:       {len(labels) - sum(labels)}")

    X, y = load_dataset(wav_paths, labels)

    model = build_model_v8()
    model.fit(X, y, batch_size=64, epochs=30, validation_split=0.2)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[OK] V8 Modell gespeichert: {MODEL_PATH}")


if __name__ == "__main__":
    main()
