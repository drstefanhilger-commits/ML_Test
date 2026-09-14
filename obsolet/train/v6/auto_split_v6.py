#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTO SPLIT V6
Teilt alle Drohnen in TRAIN und TEST auf.

Regeln:
- Quelle: data/train_48k/drone_real/
- Ziel:   drone_real_train/ (90%)
          drone_real_test/  (10%)

- Keine Dateien werden gelöscht.
- Split ist deterministisch (alphabetische Sortierung).
"""

import shutil
from pathlib import Path

SRC = Path("data/train_48k/drone_real")
DST_TRAIN = Path("data/train_48k/drone_real_train")
DST_TEST  = Path("data/train_48k/drone_real_test")

def main():
    print("======================================")
    print("           AUTO SPLIT V6 START")
    print("======================================")

    # Ordner anlegen
    DST_TRAIN.mkdir(parents=True, exist_ok=True)
    DST_TEST.mkdir(parents=True, exist_ok=True)

    # Alle Drohnen laden
    files = sorted(SRC.glob("*.wav"))
    total = len(files)

    if total == 0:
        print("[ERR] Keine Drohnen gefunden in:", SRC)
        return

    print(f"[INFO] Gefundene Drohnen: {total}")

    # 90% Train, 10% Test
    split_index = max(1, int(total * 0.9))

    train_files = files[:split_index]
    test_files  = files[split_index:]

    print(f"[INFO] TRAIN: {len(train_files)} Dateien")
    print(f"[INFO] TEST:  {len(test_files)} Dateien")

    # Dateien verschieben
    for f in train_files:
        shutil.move(str(f), DST_TRAIN / f.name)
        print(f"[TRAIN] {f.name}")

    for f in test_files:
        shutil.move(str(f), DST_TEST / f.name)
        print(f"[TEST]  {f.name}")

    print("======================================")
    print("           AUTO SPLIT V6 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
