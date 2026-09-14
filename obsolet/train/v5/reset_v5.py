#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RESET_v5 — Vollständiger Pipeline-Reset (Pfad‑B)
Löscht:
    - Modelle
    - Features
    - Scaler
    - Alte Trainingsartefakte

Startet danach:
    - build_features_v5.py
    - feature_checker_v5.py
    - train_v5.py
    - true_test_v5.py

Alles deterministisch, reproduzierbar, ohne Hidden States.
"""

import os
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------
# 1. Pfade
# ---------------------------------------------------------

MODEL_DIR = Path("models/binary")
FEATURE_DIR = Path("data/features")
FEATURE_V5_DIR = Path("data/features_v5")
SCALER_V5 = MODEL_DIR / "scaler_v5.json"

SCRIPTS = {
    "build_features_v5": "app/train/build_features_v5.py",
    "feature_checker_v5": "app/train/feature_checker_v5.py",
    "train_v5": "app/train/train_v5.py",
    "true_test_v5": "app/train/true_test_v5.py",
}

# ---------------------------------------------------------
# 2. Hilfsfunktionen
# ---------------------------------------------------------

def delete_path(path):
    if path.exists():
        print(f"[DEL] {path}")
        shutil.rmtree(path)
    else:
        print(f"[SKIP] {path} existiert nicht.")

def run_script(script):
    print(f"\n[RUN] {script}")
    subprocess.run(["python3", script], check=True)

# ---------------------------------------------------------
# 3. RESET
# ---------------------------------------------------------

def reset_all():
    print("======================================")
    print("           RESET v5 START")
    print("======================================")

    # Modelle löschen
    delete_path(MODEL_DIR)

    # Features löschen
    delete_path(FEATURE_DIR)
    delete_path(FEATURE_V5_DIR)

    # Scaler löschen
    if SCALER_V5.exists():
        print(f"[DEL] {SCALER_V5}")
        SCALER_V5.unlink()
    else:
        print(f"[SKIP] {SCALER_V5} existiert nicht.")

    print("\n======================================")
    print("        RESET v5 — DONE")
    print("======================================")

# ---------------------------------------------------------
# 4. Pipeline neu starten
# ---------------------------------------------------------

def run_pipeline():
    print("\n======================================")
    print("        PIPELINE v5 START")
    print("======================================")

    run_script(SCRIPTS["build_features_v5"])
    run_script(SCRIPTS["feature_checker_v5"])
    run_script(SCRIPTS["train_v5"])
    run_script(SCRIPTS["true_test_v5"])

    print("\n======================================")
    print("        PIPELINE v5 DONE")
    print("======================================")

# ---------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    reset_all()
    # run_pipeline()
