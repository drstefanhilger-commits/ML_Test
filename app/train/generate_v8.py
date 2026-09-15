#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from pathlib import Path

from app.sds.sds_drone_generator import make_sds_drone_frame

# ---------------------------------------------
# V8 OUTPUT DIRECTORY STRUCTURE
# ---------------------------------------------
BASE = Path("data/train_v8")
DRONE_SIM = BASE / "drone_sim"
NO_DRONE_SIM = BASE / "no_drone_sim"

for p in [DRONE_SIM, NO_DRONE_SIM]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------
# PARAMETERS FOR DRONE SIMULATION
# ---------------------------------------------
angles = [0, 45, 90, 135, 180, 225, 270, 315]
distances = [20, 30, 50, 75, 100, 150, 200]
noise_levels = [0.00, 0.05, 0.10, 0.20]

# ---------------------------------------------
# PARAMETERS FOR NO-DRONE SIMULATION
# ---------------------------------------------
noise_types = {
    "white": lambda n: np.random.randn(n),
    "pink": lambda n: np.random.randn(n) / np.sqrt(np.arange(1, n+1)),
    "brown": lambda n: np.cumsum(np.random.randn(n)),
    "wind": lambda n: np.random.randn(n) * 0.3,
    "rumble": lambda n: np.sin(2*np.pi*50*np.arange(n)/48000) * 0.3
}

NO_DRONE_DISTANCES = [20, 50, 100]
NO_DRONE_LEVELS = [0.05, 0.10, 0.20, 0.30]

SAMPLE_RATE = 48000
FRAME_LEN = 4096

# ---------------------------------------------
# GENERATE DRONE SIMULATION WAVs
# ---------------------------------------------
def generate_drone_sim():
    print("\n=== Generating DRONE_SIM V8 ===")

    for a in angles:
        for d in distances:
            for n in noise_levels:
                wav_path = DRONE_SIM / f"sds_drone_A{a}_D{d}_N{n:.2f}.wav"

                mic = make_sds_drone_frame(a, d, n)
                audio = mic.T  # shape: (4096, 8)

                sf.write(wav_path, audio, SAMPLE_RATE)
                print(f"[OK] {wav_path}")

    print("=== DRONE_SIM DONE ===")


# ---------------------------------------------
# GENERATE NO-DRONE SIMULATION WAVs
# ---------------------------------------------
def generate_no_drone_sim():
    print("\n=== Generating NO_DRONE_SIM V8 ===")

    for noise_name, noise_func in noise_types.items():
        for d in NO_DRONE_DISTANCES:
            for lvl in NO_DRONE_LEVELS:

                wav_path = NO_DRONE_SIM / f"sds_noise_{noise_name}_D{d}_L{lvl:.2f}.wav"

                # Generate noise
                noise = noise_func(FRAME_LEN).astype(np.float32)
                noise *= lvl

                # Duplicate to 8 channels
                audio = np.tile(noise.reshape(-1, 1), (1, 8))

                sf.write(wav_path, audio, SAMPLE_RATE)
                print(f"[OK] {wav_path}")

    print("=== NO_DRONE_SIM DONE ===")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    print("=====================================")
    print("     GENERATE V8 TRAINING WAV LIST")
    print("=====================================")

    generate_drone_sim()
    generate_no_drone_sim()

    print("\n[OK] V8 WAV GENERATION COMPLETE")
