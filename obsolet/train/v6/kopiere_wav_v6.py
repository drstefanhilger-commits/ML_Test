#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KOPIERE WAV V6 (DRONE)
Stellt die ursprüngliche Drone-Struktur wieder her:

- Löscht:  data/train_48k/drone_real/
- Kopiert: alle *.wav aus
           data/train_48k/drone_real_train/
           data/train_48k/drone_real_test/
  zurück nach:
           data/train_48k/drone_real/

Pfad-B konform, deterministisch.
"""

import shutil
from pathlib import Path

DRONE_BASE   = Path("data/train_48k/drone_real")
DRONE_TRAIN  = Path("data/train_48k/drone_real_train")
DRONE_TEST   = Path("data/train_48k/drone_real_test")

def clear_folder(folder: Path):
    if folder.exists():
        print(f"[DEL] Lösche Ordner: {folder}")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Neuer leerer Ordner: {folder}")

def copy_from(src: Path, dst: Path):
    if not src.exists():
        print(f"[WARN] Quelle existiert nicht: {src}")
        return

    wavs = sorted(src.glob("*.wav"))
    if not wavs:
        print(f"[WARN] Keine WAV-Dateien in: {src}")
        return

    print(f"[INFO] Kopiere {len(wavs)} WAV-Dateien aus {src}...")
    for w in wavs:
        shutil.copy(w, dst / w.name)
        print(f"[COPY] {w.name}")

def main():
    print("======================================")
    print("        KOPIERE WAV V6 START")
    print("======================================")

    # Zielordner komplett leeren
    clear_folder(DRONE_BASE)

    # Aus TRAIN kopieren
    copy_from(DRONE_TRAIN, DRONE_BASE)

    # Aus TEST kopieren
    copy_from(DRONE_TEST, DRONE_BASE)

    print("======================================")
    print("        KOPIERE WAV V6 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
