#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHECK V8 (Pfad‑B)

Prüft:
- Anzahl der WAV-Dateien in allen V8-Ordnern
- Sample-Rates der WAV-Dateien
- Kanalzahl (8-Kanal SDS)
- Länge der WAV-Dateien (4096 Samples)
- Feature-Dimension der V8-Feature-Pipeline (10243)
- DSP-Parameter
- Modell-Input-Dimension (10243)

Ziel:
Sicherstellen, dass die V8-Pipeline korrekt erzeugte 48kHz-Daten nutzt.
"""

import numpy as np
import soundfile as sf
import tensorflow as tf
from pathlib import Path

from app.train.features_v8 import extract_features_v8

# ---------------------------------------------------------
# WAV-Verzeichnisse (V8)
# ---------------------------------------------------------
WAV_DIRS = {
    "drone_real": Path("data/train_48k/drone_real"),
    "no_drone_real": Path("data/train/no_drone"),
    "drone_sim": Path("data/train_v8/drone_sim"),
    "no_drone_sim": Path("data/train_v8/no_drone_sim"),
}

# DSP-Parameter
SAMPLE_RATE = 48000
FRAME_LEN = 4096
CHANNELS = 8

# Modell
MODEL_PATH = Path("models/v8/model_binary_fp32_v8.h5")
EXPECTED_INPUT_DIM = 10243


# ---------------------------------------------------------
# Count WAV files
# ---------------------------------------------------------
def check_counts():
    print("===================================")
    print("      FILE COUNT CHECK (WAV)")
    print("===================================")

    for name, d in WAV_DIRS.items():
        if not d.exists():
            print(f"[WARN] {name}: Verzeichnis existiert nicht: {d}")
            continue

        count = len(list(d.glob("*.wav")))
        print(f"{name:15s}: {count:4d} Dateien")

    print("===================================\n")


# ---------------------------------------------------------
# Sample-Rate, Channels, Length
# ---------------------------------------------------------
def check_wav_integrity():
    print("===================================")
    print("      WAV INTEGRITY CHECK")
    print("===================================")

    for name, d in WAV_DIRS.items():
        if not d.exists():
            print(f"[WARN] {name}: Verzeichnis existiert nicht: {d}")
            continue

        sr_set = set()
        ch_set = set()
        len_set = set()
        count = 0

        for wav in sorted(d.glob("*.wav")):
            try:
                audio, sr = sf.read(wav)
                sr_set.add(sr)
                ch_set.add(audio.shape[1] if audio.ndim > 1 else 1)
                len_set.add(audio.shape[0])
                count += 1
            except Exception as e:
                print(f"[ERR] {name}: {wav.name}: {e}")

        if count == 0:
            print(f"{name:15s}: keine WAV-Dateien gefunden")
        else:
            print(f"{name:15s}: {count:4d} Dateien")
            print(f"  Sample-Rates: {sorted(sr_set)}")
            print(f"  Channels:     {sorted(ch_set)}")
            print(f"  Lengths:      {sorted(len_set)}")

    print("===================================\n")


# ---------------------------------------------------------
# Feature-Dimension-Check
# ---------------------------------------------------------
def check_feature_dims():
    print("===================================")
    print("     FEATURE-DIM CHECK (V8)")
    print("===================================")

    # Wir testen nur eine Datei aus drone_sim
    test_dir = WAV_DIRS["drone_sim"]
    test_files = list(test_dir.glob("*.wav"))

    if not test_files:
        print("[WARN] Keine drone_sim WAVs gefunden → Feature-Test übersprungen")
        print("===================================\n")
        return

    test_file = test_files[0]
    audio, sr = sf.read(test_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    feat = extract_features_v8(audio)
    print(f"Feature-Dimension: {feat.shape[0]} (erwartet: {EXPECTED_INPUT_DIM})")

    print("===================================\n")


# ---------------------------------------------------------
# DSP-Parameter-Check
# ---------------------------------------------------------
def check_dsp_params():
    print("===================================")
    print("       DSP-PARAMETER CHECK")
    print("===================================")
    print(f"SAMPLE_RATE: {SAMPLE_RATE}")
    print(f"FRAME_LEN:   {FRAME_LEN}")
    print(f"CHANNELS:    {CHANNELS}")
    print("===================================\n")


# ---------------------------------------------------------
# Model Input Check
# ---------------------------------------------------------
def check_model_input():
    print("===================================")
    print("       MODEL INPUT CHECK (V8)")
    print("===================================")

    if not MODEL_PATH.exists():
        print(f"[ERR] Modell existiert nicht: {MODEL_PATH}")
        print("===================================\n")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape[-1]

    print(f"Model Input-Dimension: {input_shape} (erwartet: {EXPECTED_INPUT_DIM})")
    print("===================================\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("===================================")
    print("              CHECK V8")
    print("===================================\n")

    check_counts()
    check_wav_integrity()
    check_feature_dims()
    check_dsp_params()
    check_model_input()

    print("===================================")
    print("              CHECK V8 DONE")
    print("===================================")


if __name__ == "__main__":
    main()
