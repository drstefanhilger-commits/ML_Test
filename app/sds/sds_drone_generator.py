#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import soundfile as sf

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

def generate_rotor_source(f0_hz=180.0,
                          harmonics=(2,3,4,5,6,7),
                          duration_samples=SDS_FRAME_LEN):

    t = np.arange(duration_samples) / SDS_SAMPLE_RATE
    src = np.sin(2 * np.pi * f0_hz * t)

    for h in harmonics:
        amp = 1.0 / h
        src += amp * np.sin(2 * np.pi * f0_hz * h * t)

    mod = 0.1 * np.sin(2 * np.pi * 2.0 * t)
    src *= (1.0 + mod)

    src = src.astype(np.float32)
    src *= 0.02
    # src /= np.max(np.abs(src) + 1e-9)

    return src

def apply_absorption_on_src(src, distance_m):
    """Einfache atmosphärische Absorption direkt auf der Rotorquelle."""
    fft = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(len(src), 1.0 / SDS_SAMPLE_RATE)

#    absorption = np.exp(-distance_m * (freqs / 2000.0)**2)
    absorption = np.exp(-distance_m * (freqs / 500.0)**2)

    fft *= absorption

    out = np.fft.irfft(fft)
    return out.astype(np.float32)

def sample_with_delay(src, delay_samples, n):
    i0 = int(n - delay_samples)
    frac = (n - delay_samples) - i0

    if i0 < 0 or i0 + 1 >= len(src):
        return 0.0

    return (1 - frac) * src[i0] + frac * src[i0 + 1]

def make_sds_drone_frame(angle_deg,
                         distance_m,
                         noise,
                         f0_hz=180.0,
                         harmonics=(2,3,4,5,6,7),
                         save_wav_path=None):

    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    # Amplituden-Abschwächung (1/d^2)
    A = 1.0 / ( distance_m * distance_m * 20.0) 


    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)

    # Rotorquelle + Absorption einmalig
    src = generate_rotor_source(f0_hz=f0_hz,
                                harmonics=harmonics,
                                duration_samples=SDS_FRAME_LEN)
    src = apply_absorption_on_src(src, distance_m)

    for ch in range(SDS_NUM_MICS):
        x, y = SDS_MIC_POSITIONS[ch]
        proj = x * dx + y * dy
        tau = proj / c
        delay_samples = tau * SDS_SAMPLE_RATE

        for n in range(SDS_FRAME_LEN):
            s = sample_with_delay(src, delay_samples, n)
            v = s

            # Noise NICHT mit Distanz skalieren
            if noise > 0.0:
                v += (np.random.randn() * 5.0) * noise

            # Distanz-Abschwächung
            v *= A 

            mic[ch][n] = v

    if save_wav_path is not None:
        sf.write(save_wav_path, mic.T, SDS_SAMPLE_RATE)
        print(f"[OK] WAV gespeichert: {save_wav_path}")

    return mic
