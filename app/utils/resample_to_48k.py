#!/usr/bin/env python3
import os
from pathlib import Path
import soundfile as sf
import librosa

TARGET_SR = 48000

DIRS = [
    "data/train/drone_real",
    "data/train/drone_sim",
    "data/train/no_drone",
    "test/true_test",
    "test/false_test",
]

for d in DIRS:
    p = Path(d)
    print(f"Resampling: {p}")

    for wav in sorted(p.glob("*.wav")):
        y, sr = librosa.load(wav, sr=TARGET_SR)
        sf.write(wav, y, TARGET_SR)
