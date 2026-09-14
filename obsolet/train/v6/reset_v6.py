#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CRESET V6
Löscht alle Feature-Ordner von V6 und legt sie leer neu an.
WAV-Daten und Modelle bleiben vollständig erhalten.
"""

import shutil
from pathlib import Path

FEATURE_ROOT = Path("data/features_v6")

def main():
    print("======================================")
    print("            CRESET V6 START")
    print("======================================")

    if FEATURE_ROOT.exists():
        print(f"[INFO] Lösche Ordner: {FEATURE_ROOT}")
        shutil.rmtree(FEATURE_ROOT)

    print("[INFO] Erzeuge leeren Feature-Root:", FEATURE_ROOT)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)

    # Leere Unterordner neu anlegen
    for sub in [
        "drone_real_train",
        "no_drone",
        "true_test",
        "false_test"
    ]:
        folder = FEATURE_ROOT / sub
        folder.mkdir(exist_ok=True)
        print(f"[OK] Neuer Ordner: {folder}")

    print("======================================")
    print("            CRESET V6 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
