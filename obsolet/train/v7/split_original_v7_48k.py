#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPLIT ORIGINAL V7 (RESAMPLE TO 48 kHz) — WSL2 VERSION
Nimmt alle Original-WAV-Dateien und resampled sie auf 48 kHz,
falls sie eine andere Sample-Rate haben.
"""

import shutil
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa

# Original-Quellen unter WSL2
DRONE_SRC = Path(
    "/mnt/c/Users/310004/Documents/Projects/Drone_Notebook/data/raw/"
    "DroneAudioDataset-master/Binary_Drone_Audio/yes_drone"
)

NODRONE_SRC = Path(
    "/mnt/c/Users/310004/Documents/Projects/Drone_Notebook/data/ESC-50/audio"
)

# Zielordner (Linux-Pfade)
DRONE_TRAIN = Path("data/train_48k/drone_real_train")
DRONE_TEST  = Path("data/train_48k/drone_real_test")

NODRONE_TRAIN = Path("data/train_48k/no_drone_train")
NODRONE_TEST  = Path("data/train_48k/no_drone_test")

TARGET_SR = 48000

def count_files(path: Path):
    wavs = list(path.glob("*.wav"))
    print(f"[COUNT] {path}: {len(wavs)} Dateien")
    return len(wavs)


def clear_and_make(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def load_and_resample(path: Path):
    """Lädt WAV und resampled auf 48 kHz, falls nötig."""
    try:
        data, sr = sf.read(path)
    except Exception as e:
        print(f"[ERR] {path.name}: {e}")
        return None

    if sr != TARGET_SR:
        print(f"[RS]  {path.name}: {sr} Hz → 48 kHz")
        data = librosa.resample(
            data.astype(np.float32),
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    return data.astype(np.float32)


def prepare_files(src: Path):
    """Liest WAVs, resampled sie und gibt eine Liste zurück."""
    wavs = sorted(src.glob("*.wav"))
    prepared = []

    if not wavs:
        print(f"[ERR] Keine WAV-Dateien gefunden in: {src}")
        return prepared

    for w in wavs:
        data = load_and_resample(w)
        if data is None:
            continue
        prepared.append((w.name, data))

    return prepared


def split_and_save(files, dst_train: Path, dst_test: Path):
    total = len(files)
    if total == 0:
        print(f"[ERR] Keine WAV-Dateien nach Resampling verfügbar.")
        return

    split_idx = max(1, int(total * 0.9))

    train_files = files[:split_idx]
    test_files  = files[split_idx:]

    print(f"[INFO] Gesamt WAVs nach Resampling: {total}")
    print(f"[INFO] TRAIN: {len(train_files)}")
    print(f"[INFO] TEST:  {len(test_files)}")

    for name, data in train_files:
        sf.write(dst_train / name, data, TARGET_SR)
        print(f"[TRAIN] {name}")

    for name, data in test_files:
        sf.write(dst_test / name, data, TARGET_SR)
        print(f"[TEST]  {name}")


def main():
    print("======================================")
    print("  SPLIT ORIGINAL V7 (RESAMPLE 48kHz) START")
    print("======================================")

    clear_and_make(DRONE_TRAIN)
    clear_and_make(DRONE_TEST)
    clear_and_make(NODRONE_TRAIN)
    clear_and_make(NODRONE_TEST)

    print("\n[INFO] Anzahl Original-Dateien:")
    count_files(DRONE_SRC)
    count_files(NODRONE_SRC)

    print("\n[INFO] Verarbeite DRONE WAVs...")
    drone_files = prepare_files(DRONE_SRC)
    split_and_save(drone_files, DRONE_TRAIN, DRONE_TEST)

    print("\n[INFO] Verarbeite NO-DRONE WAVs...")
    nodrone_files = prepare_files(NODRONE_SRC)
    split_and_save(nodrone_files, NODRONE_TRAIN, NODRONE_TEST)


    print("======================================")
    print("  SPLIT ORIGINAL V7 (RESAMPLE 48kHz) DONE")
    print("======================================")


if __name__ == "__main__":
    main()
