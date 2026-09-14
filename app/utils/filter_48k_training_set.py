#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filtert ALLE Trainingsdaten so, dass NUR 48 kHz WAVs übrig bleiben.
Erzeugt zusätzlich eine Liste aller 48 kHz Dateien.
"""

import soundfile as sf
from pathlib import Path
import shutil

TARGET_SR = 48000

TRAIN_DIRS = {
    "drone_real": Path("data/train/drone_real"),
    "drone_sim": Path("data/train/drone_sim"),
    "no_drone": Path("data/train/no_drone"),
}

OUTPUT_DIR = Path("data/train_48k")
OUTPUT_DIR.mkdir(exist_ok=True)

for name, d in TRAIN_DIRS.items():
    out = OUTPUT_DIR / name
    out.mkdir(exist_ok=True)

    print(f"\n=== Prüfe {name} ===")

    for wav in sorted(d.glob("*.wav")):
        try:
            info = sf.info(wav)
            if info.samplerate == TARGET_SR:
                print(f"48kHz: {wav.name}")
                shutil.copy(wav, out / wav.name)
        except Exception as e:
            print(f"[ERR] {wav.name}: {e}")

print("\n===================================")
print("   FILTER DONE — NUR 48 kHz benutzt")
print("===================================")
