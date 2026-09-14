#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Migration der Trainingsdaten in eine neue saubere Umgebung.

Neue Struktur (flach):
data/train/drone/
data/train/drone_sim/
data/train/no_drone/
data/train/drone_hard/
data/train/no_drone_hard/
data/train/true_test/
data/train/false_test/

Source-Daten:
Drone        = data/selected/
Drone Sim    = data/Simulation/class_A ... class_E
No-Drone     = data/ESC-50/audio/
True-Test    = test/true_test/
False-Test   = data/ESC-50/audio/  (ESC-50 wird als False-Test genutzt)
"""

import shutil
from pathlib import Path

BASE = Path("data/train")

TARGETS = [
    "drone",
    "drone_sim",
    "no_drone",
    "drone_hard",
    "no_drone_hard",
    "true_test",
    "false_test"
]

# Source paths (angepasst!)
SRC_DRONE = Path("data/selected")
SRC_SIM_ROOT = Path("data/Simulation")
SRC_NO_DRONE = Path("data/ESC-50/audio")

SRC_TRUE = Path("test/true_test")
SRC_FALSE = Path("data/ESC-50/audio")   # ESC-50 als False-Test

SRC_DRONE_HARD = Path("data/train/drone_hard")
SRC_NO_DRONE_HARD = Path("data/train/no_drone_hard")


def delete_old_structure():
    if BASE.exists():
        print("[INFO] Lösche alte Trainingsdaten-Struktur:", BASE)
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True, exist_ok=True)


def create_new_structure():
    print("[INFO] Erzeuge neue flache Trainingsdaten-Struktur...")
    for t in TARGETS:
        (BASE / t).mkdir(parents=True, exist_ok=True)


def copy_all(src, dst):
    if not src.exists():
        print("[WARN] Quelle fehlt:", src)
        return
    for wav in src.glob("*.wav"):
        shutil.copy(wav, dst / wav.name)
        print("  [+]", wav.name)


def migrate_drone():
    print("\n[INFO] Migriere Drone-Daten...")
    copy_all(SRC_DRONE, BASE / "drone")


def migrate_drone_sim():
    print("\n[INFO] Migriere Drone-Simulationen...")
    for cls in SRC_SIM_ROOT.glob("class_*"):
        copy_all(cls, BASE / "drone_sim")


def migrate_no_drone():
    print("\n[INFO] Migriere No-Drone-Daten (ESC-50)...")
    copy_all(SRC_NO_DRONE, BASE / "no_drone")


def migrate_true_test():
    print("\n[INFO] Migriere TRUE-TEST...")
    copy_all(SRC_TRUE, BASE / "true_test")


def migrate_false_test():
    print("\n[INFO] Migriere FALSE-TEST (ESC-50)...")
    copy_all(SRC_FALSE, BASE / "false_test")


def migrate_hard_samples():
    print("\n[INFO] Migriere Hard-Samples...")
    copy_all(SRC_DRONE_HARD, BASE / "drone_hard")
    copy_all(SRC_NO_DRONE_HARD, BASE / "no_drone_hard")


def run_migration():
    delete_old_structure()
    create_new_structure()

    migrate_drone()
    migrate_drone_sim()
    migrate_no_drone()
    migrate_true_test()
    migrate_false_test()
    migrate_hard_samples()

    print("\n[OK] Migration abgeschlossen.")
    print("Neue Struktur unter:", BASE)


if __name__ == "__main__":
    run_migration()
