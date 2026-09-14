#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Listet alle WAV-Dateien mit exakt 48 kHz Sample-Rate.
Getrennt nach Drohne / No-Drone.
"""

import soundfile as sf
from pathlib import Path

DIRS = {
    "train_drone_real": Path("data/train/drone_real"),
    "train_drone_sim": Path("data/train/drone_sim"),
    "train_no_drone": Path("data/train/no_drone"),
}

TARGET_SR = 48000

def list_48k():
    print("===================================")
    print("   WAV-Dateien mit 48 kHz")
    print("===================================\n")

    for name, d in DIRS.items():
        print(f"=== {name} ===")
        if not d.exists():
            print(f"[WARN] Verzeichnis existiert nicht: {d}\n")
            continue

        files_48k = []

        for wav in sorted(d.glob("*.wav")):
            try:
                info = sf.info(wav)
                if info.samplerate == TARGET_SR:
                    files_48k.append(wav.name)
            except Exception as e:
                print(f"[ERR] {wav.name}: {e}")

        if len(files_48k) == 0:
            print("Keine 48 kHz Dateien gefunden.\n")
        else:
            for f in files_48k:
                print(f" - {f}")
            print(f"\nAnzahl: {len(files_48k)}\n")

    print("===================================")
    print("   DONE")
    print("===================================")


if __name__ == "__main__":
    list_48k()
