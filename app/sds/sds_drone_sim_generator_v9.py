#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import soundfile as sf
from pathlib import Path
import shutil

SDS_NUM_MICS = 8
SDS_SAMPLE_RATE = 48000
SDS_FRAME_LEN = 4096
c = 343.0

SDS_MIC_POSITIONS = np.array([
    [ 0.03,  0.03],
    [ 0.03, -0.03],
    [-0.03,  0.03],
    [-0.03, -0.03],
    [ 0.05,  0.00],
    [-0.05,  0.00],
    [ 0.00,  0.05],
    [ 0.00, -0.05]
])

# ---------------------------------------------------------
# REALISTIC ROTOR SOURCE (V9)
# ---------------------------------------------------------
def generate_rotor_source_v9(f0_hz=180.0,
                             harmonics=(2,3,4,5,6,7),
                             duration_samples=SDS_FRAME_LEN):

    # Rotor speed jitter ±1 Hz
    f0_hz = f0_hz + np.random.normal(0.0, 1.0)

    # Harmonic jitter ±0.5 %
    harmonics = [h + np.random.normal(0.0, 0.02) for h in harmonics]

    t = np.arange(duration_samples) / SDS_SAMPLE_RATE
    src = np.sin(2 * np.pi * f0_hz * t)

    for h in harmonics:
        amp = 1.0 / h
        src += amp * np.sin(2 * np.pi * f0_hz * h * t)

    # Amplitude modulation ±3 %
    mod = 0.03 * np.sin(2 * np.pi * 2.0 * t)
    src *= (1.0 + mod)

    src = src.astype(np.float32)
    src *= 0.02

    return src


# ---------------------------------------------------------
# REALISTIC ABSORPTION
# ---------------------------------------------------------
def apply_absorption_v9(src, distance_m):
    fft = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(len(src), 1.0 / SDS_SAMPLE_RATE)

    absorption = np.exp(-distance_m * (freqs / 800.0)**2)
    fft *= absorption

    out = np.fft.irfft(fft)
    return out.astype(np.float32)


# ---------------------------------------------------------
# DELAY SAMPLING
# ---------------------------------------------------------
def sample_with_delay(src, delay_samples, n):
    i0 = int(n - delay_samples)
    frac = (n - delay_samples) - i0

    if i0 < 0 or i0 + 1 >= len(src):
        return 0.0

    return (1 - frac) * src[i0] + frac * src[i0 + 1]


# ---------------------------------------------------------
# MAIN V9 SIMULATION FUNCTION
# ---------------------------------------------------------
def make_sds_drone_frame_v9(angle_deg,
                            distance_m,
                            noise,
                            f0_hz=180.0,
                            harmonics=(2,3,4,5,6,7),
                            save_wav_path=None):

    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    # Realistic distance attenuation
    A = 1.0 / (distance_m**1.7 * 15.0)

    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)

    # Rotor source
    src = generate_rotor_source_v9(f0_hz=f0_hz,
                                   harmonics=harmonics,
                                   duration_samples=SDS_FRAME_LEN)

    src = apply_absorption_v9(src, distance_m)

    # Channel imbalance ±2 %
    channel_gain = 1.0 + np.random.normal(0.0, 0.02, size=SDS_NUM_MICS)

    # Phase noise ±0.3 samples
    phase_noise = np.random.normal(0.0, 0.3, size=SDS_NUM_MICS)

    # Microphone noise floor
    base_noise = 0.001

    for ch in range(SDS_NUM_MICS):
        x, y = SDS_MIC_POSITIONS[ch]
        proj = x * dx + y * dy
        tau = proj / c

        delay_samples = tau * SDS_SAMPLE_RATE + phase_noise[ch]

        for n in range(SDS_FRAME_LEN):
            s = sample_with_delay(src, delay_samples, n)

            v = s * A

            # Noise floor
            v += np.random.randn() * base_noise

            # Environmental noise
            if noise > 0.0:
                v += np.random.randn() * noise * 0.5

            # Channel imbalance
            v *= channel_gain[ch]

            mic[ch][n] = v

    if save_wav_path is not None:
        sf.write(save_wav_path, mic.T, SDS_SAMPLE_RATE)

    return mic


# ---------------------------------------------------------
# TEST-LAUF: DELETE OLD FILES + GENERATE NEW
# ---------------------------------------------------------
def main():

    out_dir = Path("data/test_sim_v9/drone_sim")
    print("=== V9 Simulation Test-Lauf ===")

    # 1) Delete old folder
    if out_dir.exists():
        print("[INFO] Lösche alten Test-Ordner...")
        shutil.rmtree(out_dir)

    # 2) Create new folder
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = [0, 45, 90, 135, 180]
    distances = [50, 100, 150]
    noises = [0.00, 0.05, 0.10]

    # 3) Generate new simulations
    for a in angles:
        for d in distances:
            for n in noises:
                path = out_dir / f"sim_A{a}_D{d}_N{n:.2f}.wav"
                make_sds_drone_frame_v9(a, d, n, save_wav_path=str(path))
                print(f"[OK] {path}")

    print("\n[FINISHED] V9 Simulation Test-Lauf abgeschlossen.")


if __name__ == "__main__":
    main()
