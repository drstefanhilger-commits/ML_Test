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
# REALISTIC NO-DRONE NOISE SOURCES (V9)
# ---------------------------------------------------------

def noise_white(level=0.02):
    return np.random.randn(SDS_FRAME_LEN).astype(np.float32) * level

def noise_pink(level=0.02):
    freqs = np.fft.rfftfreq(SDS_FRAME_LEN, 1.0 / SDS_SAMPLE_RATE)
    spectrum = np.random.randn(len(freqs)) / (freqs + 1e-6)
    noise = np.fft.irfft(spectrum).astype(np.float32)
    noise *= level
    return noise

def noise_wind_v9_turbulent(level=0.03):
    freqs = np.fft.rfftfreq(SDS_FRAME_LEN, 1.0 / SDS_SAMPLE_RATE)
    spectrum = np.random.randn(len(freqs)) / (freqs + 1e-3)
    wind = np.fft.irfft(spectrum).astype(np.float32)

    gusts = np.random.randn(SDS_FRAME_LEN).astype(np.float32)
    gusts = np.convolve(gusts, np.ones(50)/50, mode='same')
    gusts *= 0.3

    rumble_freq = np.random.uniform(5, 20)
    t = np.arange(SDS_FRAME_LEN) / SDS_SAMPLE_RATE
    rumble = np.random.randn() * 0.1 * np.sin(2 * np.pi * rumble_freq * t)
    rumble = rumble.astype(np.float32)

    noise = wind + gusts + rumble
    noise /= np.max(np.abs(noise) + 1e-9)
    noise *= level

    return noise.astype(np.float32)

def noise_impulse_v9(level=0.03):
    noise = np.random.randn(SDS_FRAME_LEN).astype(np.float32) * (level * 0.3)

    num_bursts = np.random.randint(3, 8)
    for _ in range(num_bursts):
        pos = np.random.randint(0, SDS_FRAME_LEN)
        width = np.random.randint(20, 80)
        burst = np.random.randn(width).astype(np.float32)
        burst *= np.random.uniform(0.1, 0.3)

        end = min(pos + width, SDS_FRAME_LEN)
        noise[pos:end] += burst[:end-pos]

    noise = np.convolve(noise, np.ones(10)/10, mode='same')

    noise /= np.max(np.abs(noise) + 1e-9)
    noise *= level

    return noise.astype(np.float32)

def noise_lowfreq(level=0.03):
    t = np.arange(SDS_FRAME_LEN) / SDS_SAMPLE_RATE
    lf = np.sin(2 * np.pi * np.random.uniform(20, 80) * t).astype(np.float32)
    lf *= level
    return lf

# ---------------------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------------------

def make_no_drone_frame_v9(noise_type="white", level=0.03):
    if noise_type == "white":
        src = noise_white(level)
    elif noise_type == "pink":
        src = noise_pink(level)
    elif noise_type == "wind":
        src = noise_wind_v9_turbulent(level)
    elif noise_type == "impulse":
        src = noise_impulse_v9(level)  # <<< FIX AKTIV
    elif noise_type == "lowfreq":
        src = noise_lowfreq(level)
    else:
        src = noise_white(level)

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

def main():

    out_dir = Path("data/test_sim_v9/no_drone_sim")
    print("=== V9 No-Drone Simulation Generator (Wind + Impulse FIX) ===")

    if out_dir.exists():
        print("[INFO] Lösche alten no_drone_sim Ordner...")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    noise_types = ["white", "pink", "wind", "impulse", "lowfreq"]
    levels = [0.02, 0.03, 0.05]

    idx = 0
    for nt in noise_types:
        for lvl in levels:
            mic = make_no_drone_frame_v9(nt, lvl)
            path = out_dir / f"no_drone_{nt}_L{lvl:.2f}_{idx}.wav"
            sf.write(path, mic.T, SDS_SAMPLE_RATE)
            print(f"[OK] {path}")
            idx += 1

    print("\n[FINISHED] V9 No-Drone Simulation (Wind + Impulse FIX) abgeschlossen.")


if __name__ == "__main__":
    main()
