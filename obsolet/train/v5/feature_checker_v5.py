#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FEATURE-CHECKER V5 (GLOBAL NORMALIZATION)
Prüft ALLE Features nach build_features_v5:

- Existenz & Anzahl der Features
- Feature-Dimension
- Globale Normierungsparameter
- Statistik der unnormalisierten Features
- Statistik der normalisierten Features
- Prüfung TRUE-TEST & FALSE-TEST gegen Trainingsbereich
"""

import os
import json
import numpy as np
from pathlib import Path

FEATURE_DIR = Path("data/features_v5")
SCALER_JSON = Path("models/binary/scaler_v5.json")

CATEGORIES = [
    "drone_real",
    "drone_sim",
    "no_drone",
    "true_test",
    "false_test"
]

def load_scaler():
    with open(SCALER_JSON, "r") as fp:
        d = json.load(fp)
    mean = np.array(d["mean"], dtype=np.float32)
    scale = np.array(d["scale"], dtype=np.float32)
    return mean, scale

def load_features(category):
    folder = FEATURE_DIR / category
    feats = []

    for f in sorted(folder.glob("*.npy")):
        feat = np.load(f)
        feats.append(feat)

    return np.array(feats, dtype=np.float32)

def print_stats(name, feats, mean, scale):
    print(f"\n=== {name} ===")
    print(f"Anzahl Features: {len(feats)}")

    if len(feats) == 0:
        print("WARNUNG: Keine Features gefunden!")
        return

    # Dimension prüfen
    if feats.shape[1] != 40:
        print(f"FEHLER: Feature-Dimension ist {feats.shape[1]}, erwartet 40!")
    else:
        print("Feature-Dimension: OK (40)")

    # Unnormierte Statistik
    print("\n--- Unnormierte Statistik ---")
    print(f"Mean: {np.mean(feats):.3f}")
    print(f"Std:  {np.std(feats):.3f}")
    print(f"Min:  {np.min(feats):.3f}")
    print(f"Max:  {np.max(feats):.3f}")

    # Normierte Statistik
    feats_norm = (feats - mean) / scale

    print("\n--- Normierte Statistik ---")
    print(f"Mean: {np.mean(feats_norm):.3f}")
    print(f"Std:  {np.std(feats_norm):.3f}")
    print(f"Min:  {np.min(feats_norm):.3f}")
    print(f"Max:  {np.max(feats_norm):.3f}")

    # Prüfung: liegen Features im Trainingsbereich?
    train_min = -3.0
    train_max = 7.5

    out_of_range = np.sum((feats_norm < train_min) | (feats_norm > train_max))

    print(f"\nOut-of-range Werte: {out_of_range}")

    if out_of_range > 0:
        print("WARNUNG: TRUE/FALSE-TEST Features liegen außerhalb des Trainingsbereichs!")
    else:
        print("OK: TRUE/FALSE-TEST Features liegen im Trainingsbereich.")

def main():
    print("===================================")
    print("     FEATURE-CHECKER V5 START")
    print("===================================\n")

    # Scaler laden
    mean, scale = load_scaler()
    print("[OK] Scaler geladen")

    # Alle Kategorien prüfen
    for cat in CATEGORIES:
        feats = load_features(cat)
        print_stats(cat, feats, mean, scale)

    print("\n===================================")
    print("     FEATURE-CHECKER V5 DONE")
    print("===================================\n")

if __name__ == "__main__":
    main()
