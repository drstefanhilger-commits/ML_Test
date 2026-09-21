import os
import numpy as np
import soundfile as sf

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SR = 48000
DURATION = 2.0
N_SAMPLES = 160

# Rotorharmonik-Parameter
F0_MIN = 80      # minimale Blade-Pass-Frequency
F0_MAX = 120     # maximale Blade-Pass-Frequency
N_HARM = 12      # Anzahl der Harmoniken

# Amplitudenverhältnisse (realistisch)
HARM_RATIO = np.array([1.0, 0.55, 0.35, 0.25, 0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03])

# AM-Modulation (RPM-Jitter)
AM_DEPTH = 0.15
AM_RATE = 4.0

# Noise-Floor
NOISE_LEVEL = 0.02

# Distanz-Dämpfung
DIST_MIN = 5.0
DIST_MAX = 40.0


def generate_rotor_sound():
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

    # zufällige Grundfrequenz
    f0 = np.random.uniform(F0_MIN, F0_MAX)

    # zufällige Distanz
    dist = np.random.uniform(DIST_MIN, DIST_MAX)
    dist_gain = 1.0 / dist

    # AM-Modulation
    am = 1.0 + AM_DEPTH * np.sin(2 * np.pi * AM_RATE * t)

    # Harmoniken erzeugen
    signal = np.zeros_like(t)
    for k in range(1, N_HARM + 1):
        fk = k * f0
        ak = HARM_RATIO[k - 1] * dist_gain

        # leichte Frequenzmodulation (RPM-Jitter)
        jitter = 1.0 + 0.01 * np.sin(2 * np.pi * (0.3 * k) * t)

        signal += ak * np.sin(2 * np.pi * fk * jitter * t)

    # Noise-Floor
    noise = NOISE_LEVEL * np.random.randn(len(t))

    return signal + noise


def main():
    OUT_DIR = "./data/train_48k/sim_drone_test"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Erzeuge {N_SAMPLES} simulierte Drohnen in {OUT_DIR}")

    for i in range(N_SAMPLES):
        audio = generate_rotor_sound()
        audio /= np.max(np.abs(audio)) + 1e-9  # normalisieren

        path = os.path.join(OUT_DIR, f"sim_drone_{i:03d}.wav")
        sf.write(path, audio, SR)

    print("[INFO] Fertig. Simulierte Drohnen erzeugt.")


if __name__ == "__main__":
    main()
