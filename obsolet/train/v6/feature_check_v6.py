#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FEATURE CHECK V6
Prüft alle erzeugten Features in data/features_v6:
- Dimensionen
- NaNs / Infs
- Wertebereiche
- Konsistenz zwischen TRAIN und TEST
- Scaler-Kompatibilität
"""

import numpy as np
import json
from pathlib import Path

FEATURE_ROOT = Path("data/features_v6")
SCALER_JSON = Path("models/binary/scaler_v6.json")

GROUPS = [
    "drone_real_train",
    "no_drone",
    "true_test",
    "false_test"
]

def load_scaler():
    if not SCALER_JSON.exists():
        print("[WARN] Scaler nicht gefunden:", SCALER_JSON)
        return None, None

    with open(SCALER_JSON, "r") as fp:
        d = json.load(fp)

    mean = np.array(d["mean"], dtype=np.float32)
    scale = np.array(d["scale"], dtype=np.float32)
    return mean, scale

def check_feature_file(path):
    try:
        feat = np.load(path)
    except Exception as e:
        return None, f"LOAD ERROR: {e}"

    if feat.ndim != 1:
        return None, f"BAD SHAPE: {feat.shape}"

    if np.any(np.isnan(feat)):
        return None, "NAN VALUES"

    if np.any(np.isinf(feat)):
        return None, "INF VALUES"

    return feat, None

def main():
    print("======================================")
    print("         FEATURE CHECK V6 START")
    print("======================================")

    mean, scale = load_scaler()

    dims = []
    errors = []
    stats = {}

    for group in GROUPS:
        folder = FEATURE_ROOT / group
        print(f"\n--- Gruppe: {group} ---")

        if not folder.exists():
            print(f"[WARN] Ordner fehlt: {folder}")
            continue

        feats = []
        for npy in sorted(folder.glob("*.npy")):
            feat, err = check_feature_file(npy)
            if err:
                errors.append((npy, err))
                print(f"[ERR] {npy.name}: {err}")
                continue

            feats.append(feat)
            dims.append(len(feat))

        if feats:
            feats = np.array(feats)
            stats[group] = {
                "count": len(feats),
                "dim": feats.shape[1],
                "mean": float(np.mean(feats)),
                "std": float(np.std(feats)),
                "min": float(np.min(feats)),
                "max": float(np.max(feats)),
            }

            print(f"[OK] {len(feats)} Dateien")
            print(f"     Feature-Dim: {feats.shape[1]}")
            print(f"     Mean: {stats[group]['mean']:.3f}")
            print(f"     Std:  {stats[group]['std']:.3f}")
            print(f"     Min:  {stats[group]['min']:.3f}")
            print(f"     Max:  {stats[group]['max']:.3f}")
        else:
            print("[WARN] Keine Features gefunden")

    # -----------------------------------------------------
    # Dimensionen prüfen
    # -----------------------------------------------------
    print("\n=== DIMENSION CHECK ===")
    if len(set(dims)) == 1:
        print("[OK] Alle Feature-Dimensionen konsistent:", dims[0])
    else:
        print("[ERR] Inkonsistente Dimensionen:", set(dims))

    # -----------------------------------------------------
    # Scaler-Kompatibilität prüfen
    # -----------------------------------------------------
    if mean is not None:
        print("\n=== SCALER CHECK ===")
        print(f"Scaler-Dimension: {len(mean)}")

        if len(mean) != dims[0]:
            print("[ERR] Scaler-Dimension passt NICHT zu Features!")
        else:
            print("[OK] Scaler-Dimension korrekt")

    # -----------------------------------------------------
    # Fehlerliste
    # -----------------------------------------------------
    print("\n=== ERROR SUMMARY ===")
    if errors:
        for path, err in errors:
            print(f"[ERR] {path.name}: {err}")
    else:
        print("[OK] Keine Fehler gefunden")

    print("\n======================================")
    print("         FEATURE CHECK V6 DONE")
    print("======================================")

if __name__ == "__main__":
    main()
