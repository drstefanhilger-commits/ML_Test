import os
import numpy as np
import soundfile as sf

# ---------------------------------------------------------
# Basisparameter
# ---------------------------------------------------------
SR = 48000
DURATION = 2.0
N_SAMPLES = 160

F0_MIN = 80.0
F0_MAX = 120.0
N_HARM = 12

# Fernfeld-Entfernungen (realistisch für akustische Detektion)
DIST_MIN = 50.0   # 50 m
DIST_MAX = 300.0  # 300 m

# Pegelabfall: 1/r^2
def distance_gain(r: float) -> float:
    return 1.0 / (r ** 2)


# Noise-Level: Fernfeld ist stark von Umgebung dominiert
WHITE_NOISE_LEVEL = 0.10
PINK_NOISE_LEVEL = 0.18

# Harmoniken im Fernfeld sind extrem schwach
HARMONIC_BASE_LEVEL = 0.08

# leichte Modulation, aber viel schwächer als im Nahfeld
AM_DEPTH = 0.10
AM_RATES = [2.0, 4.5]


def generate_pink_noise(n_samples: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_samples, 1.0 / SR)
    mag = 1.0 / (freqs + 1.0)
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag)))
    spec = mag * phase
    noise = np.fft.irfft(spec, n=n_samples)
    noise /= np.max(np.abs(noise)) + 1e-9
    return noise


def generate_farfield_drone():
    n_samples = int(SR * DURATION)
    t = np.linspace(0, DURATION, n_samples, endpoint=False)

    # zufällige Fernfeld-Entfernung
    dist = np.random.uniform(DIST_MIN, DIST_MAX)
    gain = distance_gain(dist)

    # Grundfrequenz
    f0 = np.random.uniform(F0_MIN, F0_MAX)

    # sehr schwache, „verwaschene“ Harmoniken
    signal = np.zeros_like(t)

    # leichte AM-Modulation (Fernfeld: kaum hörbar, aber vorhanden)
    am = np.ones_like(t)
    for rate in AM_RATES:
        phase = np.random.uniform(0, 2 * np.pi)
        am += AM_DEPTH * np.sin(2 * np.pi * rate * t + phase)
    am = np.clip(am, 0.5, None)

    # schwacher, diffuser Rotoranteil
    for k in range(1, N_HARM + 1):
        fk = k * f0

        # sehr schwache Amplitude, stark distanzgedämpft
        amp = HARMONIC_BASE_LEVEL * gain * (1.0 / (k ** 1.2))

        # zufällige Phase
        phase0 = np.random.uniform(0, 2 * np.pi)

        # leicht „wabernde“ Frequenz (Jitter), aber nicht stark
        jitter = 0.01 * np.random.randn(n_samples)
        jitter = np.convolve(jitter, np.ones(256) / 256, mode="same")
        fk_time = fk * (1.0 + jitter)

        theta = 2 * np.pi * np.cumsum(fk_time) / SR + phase0
        harmonic = np.sin(theta)

        signal += amp * am * harmonic

    # Fernfeld-Noise: rosa + weiß, dominiert klar
    white = WHITE_NOISE_LEVEL * np.random.randn(n_samples)
    pink = PINK_NOISE_LEVEL * generate_pink_noise(n_samples)

    # atmosphärische Dämpfung: Höhen stärker gedämpft (einfacher Tilt)
    tilt = np.linspace(1.0, 0.3, n_samples)
    noise = (white + pink) * tilt

    out = signal + noise
    out /= np.max(np.abs(out)) + 1e-9
    return out


def main():
    OUT_DIR = "./data/train_48k/sim_drone_test"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Lösche alte WAV-Dateien in {OUT_DIR}")
    for f in os.listdir(OUT_DIR):
        if f.lower().endswith(".wav"):
            os.remove(os.path.join(OUT_DIR, f))
    print("[INFO] Alte Dateien gelöscht.")

    print(f"[INFO] Erzeuge {N_SAMPLES} Fernfeld-Sim-Drohnen (50–300 m)")

    for i in range(N_SAMPLES):
        audio = generate_farfield_drone()
        sf.write(os.path.join(OUT_DIR, f"sim_farfield_{i:03d}.wav"), audio, SR)

    print("[INFO] Fertig.")


if __name__ == "__main__":
    main()
