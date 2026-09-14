#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DELETE ALL V6
Löscht ALLE Feature-Ordner, Modelle, Scaler und Logs für V6.

WICHTIG:
- WAV-Daten bleiben unangetastet.
- Scripts bleiben unangetastet.
- Pfad-B konform, deterministisch.

Löscht:
- data/features_v6/
- models/binary/model_binary_v6.h5
- models/binary/model_binary_v6_int8.tflite
- models/binary/scaler_v6.json
- logs/train_v6.log (falls vorhanden)
"""

from pathlib import Path
import shutil

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
FEATURE_DIR = Path("data/features_v6")
MODEL_DIR   = Path("models/binary")
LOG_DIR     = Path("logs")

FILES = [
    MODEL_DIR / "model_binary_v6.h5",
    MODEL_DIR / "model_binary_v6_int8.tflite",
    MODEL_DIR / "scaler_v6.json",
    LOG_DIR   / "train_v6.log",
]

# ---------------------------------------------------------
# DELETE HELPERS
# ---------------------------------------------------------
def delete_file(path: Path):
    if path.exists():
        print(f"[DEL] Datei: {path}")
        path.unlink()
    else:
        print(f"[SKIP] Datei existiert nicht: {path}")

def delete_folder(path: Path):
    if path.exists():
        print(f"[DEL] Ordner: {path}")
        shutil.rmtree(path)
    else:
        print(f"[SKIP] Ordner existiert nicht: {path}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("           DELETE ALL V6 START")
    print("======================================")

    # Features löschen
    delete_folder(FEATURE_DIR)

    # Modelle + Scaler löschen
    for f in FILES:
        delete_file(f)

    print("======================================")
    print("           DELETE ALL V6 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
