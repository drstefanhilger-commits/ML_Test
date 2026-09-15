#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

from sds_drone_generator import make_sds_drone_frame

MODEL_PATH = Path("models/v7/model_binary_fp32_v7.h5")

SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FIXED_FRAMES = 256

def extract_features(audio):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=20,
        fmax=20000,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    frames = mel_db.shape[1]
    if frames < FIXED_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0,FIXED_FRAMES-frames)), mode="constant")
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    return mel_db.flatten().astype(np.float32)

def classify_sds_mono_mix(angle, distance, noise):
    mic = make_sds_drone_frame(angle, distance, noise)

    # *** MONO-MIX ***
    mono = mic.mean(axis=0)

    feat = extract_features(mono)

    model = tf.keras.models.load_model(MODEL_PATH)
    score = float(model.predict(feat.reshape(1,-1))[0][0])

    label = "DRONE" if score > 0.5 else "NO DRONE"
    print(f"Angle={angle:3d}°, Dist={distance:3.1f}m, Noise={noise:.2f} → Score={score:.3f} ({label})")

# Beispieltest
classify_sds_mono_mix(angle=90, distance=2.0, noise=0.1)
