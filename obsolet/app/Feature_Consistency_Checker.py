#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature-Consistency-Checker (Pfad‑B, aktualisiert)

Prüft:
- Sample-Rates der Trainings-WAVs (data/train/…)
- Sample-Rates der Test-WAVs (test/…)
- Feature-Dimensionen der .npy-Dateien
- DSP-Parameter
- TFLite-Quantisierung

Ziel:
Sicherstellen, dass Training / True-Test / False-Test dieselben DSP-Parameter nutzen.
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf

# ---------------------------------------------------------
# WAV-Verzeichnisse (AKTUALISIERT)
# ---------------------------------------------------------
WAV_DIRS = {
    "train_drone_real": Path("data/train/drone_real"),
    "train_drone_sim": Path("data/train/drone_sim"),
    "train_no_drone": Path("data/train/no_drone"),
    "true_test": Path("test/true_test"),
    "false_test": Path("test/false_test"),
}

# ---------------------------------------------------------
# Feature-Verzeichnisse (unverändert)
# ---------------------------------------------------------
FEATURE_DIRS = {
    "drone_real": Path("data/features/drone_real"),
    "drone_sim": Path("data/features/drone_sim"),
    "no_drone": Path("data/features/no_drone"),
}

TFLITE_MODEL = Path("models/binary/model_binary_int8.tflite")

# DSP-Parameter (müssen zu build_features.py passen)
SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000


# ---------------------------------------------------------
# Sample-Rate-Check
# ---------------------------------------------------------
def check_sample_rates():
    print("===================================")
    print("      SAMPLE-RATE CHECK (WAV)")
    print("===================================")

    for name, d in WAV_DIRS.items():
        if not d.exists():
            print(f"[WARN] {name}: Verzeichnis existiert nicht: {d}")
            continue

        sr_set = set()
        count = 0

        for wav in sorted(d.glob("*.wav")):
            try:
                info = sf.info(wav)
                sr_set.add(info.samplerate)
                count += 1
            except Exception as e:
                print(f"[ERR] {name}: {wav.name}: {e}")

        if count == 0:
            print(f"{name:15s}: keine WAV-Dateien gefunden")
        else:
            srs = ", ".join(str(s) for s in sorted(sr_set))
            print(f"{name:15s}: {count:4d} Dateien, Sample-Rates: {srs}")

    print("===================================\n")


# ---------------------------------------------------------
# Feature-Dimension-Check
# ---------------------------------------------------------
def check_feature_dims():
    print("===================================")
    print("     FEATURE-DIM CHECK (.npy)")
    print("===================================")

    for name, d in FEATURE_DIRS.items():
        if not d.exists():
            print(f"[WARN] {name}: Verzeichnis existiert nicht: {d}")
            continue

        dims = set()
        count = 0

        for npy in sorted(d.glob("*.npy")):
            try:
                feat = np.load(npy)
                dims.add(feat.shape[-1])
                count += 1
            except Exception as e:
                print(f"[ERR] {name}: {npy.name}: {e}")

        if count == 0:
            print(f"{name:12s}: keine .npy-Dateien gefunden")
        else:
            dim_str = ", ".join(str(d) for d in sorted(dims))
            print(f"{name:12s}: {count:4d} Dateien, Feature-Dimension(en): {dim_str}")

    print("===================================\n")


# ---------------------------------------------------------
# DSP-Parameter-Check
# ---------------------------------------------------------
def check_dsp_params():
    print("===================================")
    print("       DSP-PARAMETER CHECK")
    print("===================================")
    print(f"SAMPLE_RATE (Python): {SAMPLE_RATE}")
    print(f"N_FFT:                {N_FFT}")
    print(f"HOP:                  {HOP}")
    print(f"N_MELS:               {N_MELS}")
    print(f"FMIN:                 {FMIN}")
    print(f"FMAX:                 {FMAX}")
    print("===================================\n")


# ---------------------------------------------------------
# TFLite-Quantization-Check
# ---------------------------------------------------------
def check_tflite_quantization():
    print("===================================")
    print("     TFLITE QUANTIZATION CHECK")
    print("===================================")

    if not TFLITE_MODEL.exists():
        print(f"[ERR] Modell existiert nicht: {TFLITE_MODEL}")
        print("===================================\n")
        return

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input dtype:      {input_details[0]['dtype']}")
    print(f"Input quant:      {input_details[0]['quantization']}")
    print(f"Output dtype:     {output_details[0]['dtype']}")
    print(f"Output quant:     {output_details[0]['quantization']}")

    print("===================================\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("===================================")
    print("   FEATURE-CONSISTENCY-CHECKER")
    print("===================================\n")

    check_sample_rates()
    check_feature_dims()
    check_dsp_params()
    check_tflite_quantization()

    print("===================================")
    print("   FEATURE-CONSISTENCY-CHECK DONE")
    print("===================================")


if __name__ == "__main__":
    main()
