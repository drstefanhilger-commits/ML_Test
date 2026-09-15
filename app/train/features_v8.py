#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import librosa

SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FIXED_FRAMES = 256   # 40 * 256 = 10240

def extract_features_v8(audio: np.ndarray) -> np.ndarray:
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

    mel_db = librosa.power_to_db(mel, ref=1.0)

    frames = mel_db.shape[1]
    if frames < FIXED_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0,FIXED_FRAMES-frames)), mode="constant")
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    feat_mel = mel_db.flatten().astype(np.float32)

    # Level-Features
    rms = np.sqrt(np.mean(audio**2)).astype(np.float32)
    peak = np.max(np.abs(audio)).astype(np.float32)

    noise_band = mel_db[-10:, :]
    noise_energy = np.mean(noise_band)
    signal_energy = np.mean(mel_db)
    snr = (signal_energy - noise_energy).astype(np.float32)

    level_feats = np.array([rms, peak, snr], dtype=np.float32)

    feat = np.concatenate([feat_mel, level_feats], axis=0)

    return feat
