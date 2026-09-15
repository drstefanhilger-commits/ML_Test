#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import soundfile as sf
from pathlib import Path
import librosa

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

def test_sds_drone(angle, distance, noise):
    wav_path = f"test/sds/sds_angle{angle}_dist{distance}_noise{noise}.wav"
    mic = make_sds_drone_frame(angle, distance, noise, save_wav_path=wav_path)

    mono = mic.mean(axis=0)
    feat = extract_features(mono)

    model = tf.keras.models.load_model(MODEL_PATH)
    score = float(model.predict(feat.reshape(1,-1))[0][0])

    print(f"Angle={angle}°, Dist={distance}m, Noise={noise} → Score={score:.3f}")

angles = [0, 45, 90, 135, 180, 225, 270, 315]
distances = [1.0, 2.0, 3.0, 5.0]
noise_levels = [0.0, 0.1, 0.2]

for a in angles:
    for d in distances:
        for n in noise_levels:
            test_sds_drone(a, d, n)
