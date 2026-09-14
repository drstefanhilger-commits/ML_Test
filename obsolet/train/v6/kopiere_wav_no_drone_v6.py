#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KOPIERE WAV NO-DRONE V6
Stellt die ursprüngliche No-Drone-Struktur wieder her:

- Löscht:  data/train_48k/no_drone/
- Kopiert: alle *.wav aus
           data/train_48k/no_drone_train/
           data/train_48k/no_drone_test/
  zurück nach:
           data/train_48k/no_drone/

Pfad-B konform, deterministisch.
"""

import shutil
from pathlib import Path

NODRONE_BASE   = Path("data/train_48k/no_drone")
NODRONE_TRAIN  = Path("data/train_48k/no_drone_train")
NODRONE_TEST   = Path("data/train_48k/no_drone_test")

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
    print("     KOPIERE WAV NO-DRONE V6 START")
    print("======================================")

    # Zielordner komplett leeren
    clear_folder(NODRONE_BASE)

    # Aus TRAIN kopieren
    copy_from(NODRONE_TRAIN, NODRONE_BASE)

    # Aus TEST kopieren
    copy_from(NODRONE_TEST, NODRONE_BASE)

    print("======================================")
    print("     KOPIERE WAV NO-DRONE V6 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
