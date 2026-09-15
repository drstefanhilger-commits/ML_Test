#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from pathlib import Path
import shutil

SDS_NUM_MICS = 8
SDS_SAMPLE_RATE = 48000
SDS_FRAME_LEN = 4096

# ---------------------------------------------------------
# SAFE NO-DRONE NOISE SOURCES (V9 SAFE)
# ---------------------------------------------------------

def noise_white(level=0.02):
    return np.random.randn(SDS_FRAME_LEN).astype(np.float32) * level

def noise_pink(level=0.02):
    freqs = np.fft.rfftfreq(SDS_FRAME_LEN, 1.0 / SDS_SAMPLE_RATE)
    spectrum = np.random.randn(len(freqs)) / (freqs + 1e-6)
    noise = np.fft.irfft(spectrum).astype(np.float32)
    noise *= level
    return noise

def noise_lowfreq(level=0.03):
    t = np.arange(SDS_FRAME_LEN) / SDS_SAMPLE_RATE
    lf = np.sin(2 * np.pi * np.random.uniform(20, 80) * t).astype(np.float32)
    lf *= level
    return lf

# ---------------------------------------------------------
# MAIN SAFE GENERATOR
# ---------------------------------------------------------

def make_no_drone_frame_v9_safe(noise_type="pink", level=0.03):
    if noise_type == "white":
        src = noise_white(level)
    elif noise_type == "pink":
        src = noise_pink(level)
    elif noise_type == "lowfreq":
        src = noise_lowfreq(level)
    else:
        src = noise_pink(level)

    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)

    channel_gain = 1.0 + np.random.normal(0.0, 0.02, size=SDS_NUM_MICS)
    noise_floor = 0.001

    for ch in range(SDS_NUM_MICS):
        v = src * channel_gain[ch]
        v += np.random.randn(SDS_FRAME_LEN).astype(np.float32) * noise_floor
        mic[ch] = v

    return mic

# ---------------------------------------------------------
# TEST-LAUF: DELETE OLD FILES + GENERATE NEW
# ---------------------------------------------------------

def main(use_no_drone_sim=True):

    out_dir = Path("data/test_sim_v9/no_drone_sim_safe")
    print("=== V9 SAFE No-Drone Simulation Generator ===")

    if not use_no_drone_sim:
        print("[INFO] Keine Berechnung – no_drone_sim deaktiviert.")
        return

    if out_dir.exists():
        print("[INFO] Lösche alten no_drone_sim_safe Ordner...")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    noise_types = ["white", "pink", "lowfreq"]
    levels = [0.02, 0.03, 0.05]

    idx = 0
    for nt in noise_types:
        for lvl in levels:
            mic = make_no_drone_frame_v9_safe(nt, lvl)
            path = out_dir / f"no_drone_safe_{nt}_L{lvl:.2f}_{idx}.wav"
            sf.write(path, mic.T, SDS_SAMPLE_RATE)
            print(f"[OK] {path}")
            idx += 1

    print("\n[FINISHED] SAFE No-Drone Simulation abgeschlossen.")


if __name__ == "__main__":
    main(use_no_drone_sim=True)
