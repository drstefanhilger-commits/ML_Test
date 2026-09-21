import os
import numpy as np
import soundfile as sf

# ---------------------------------------------------------
# SDS / Signal-Parameter
# ---------------------------------------------------------
SR = 48000
N_FFT = 4096
HOP = 2048

DURATION = 2.0
N_SAMPLES = 160

F0_MIN = 80.0
F0_MAX = 120.0
N_HARM = 12

# Bandbreite der Harmoniken (Hz)
HARM_BW_BASE = 40.0

# AM-Modulation
AM_DEPTH_BASE = 0.20
AM_RATES = [3.0, 7.0, 11.0]  # mehrere Mod-Frequenzen

# RPM-Jitter
RPM_JITTER_BASE = 0.03  # 3 %

# Noise-Floor
WHITE_NOISE_LEVEL = 0.04
PINK_NOISE_LEVEL = 0.06

# Distanz-Dämpfung
DIST_MIN = 5.0
DIST_MAX = 60.0


def generate_pink_noise(n_samples: int) -> np.ndarray:
    # einfacher 1/f-Noise-Ansatz
    freqs = np.fft.rfftfreq(n_samples, 1.0 / SR)
    mag = 1.0 / (freqs + 1.0)
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag)))
    spec = mag * phase
    noise = np.fft.irfft(spec, n=n_samples)
    noise /= np.max(np.abs(noise)) + 1e-9
    return noise


def generate_chaotic_rotor():
    n_samples = int(SR * DURATION)
    t = np.linspace(0, DURATION, n_samples, endpoint=False)

    # zufällige Grundfrequenz und Distanz
    f0 = np.random.uniform(F0_MIN, F0_MAX)
    dist = np.random.uniform(DIST_MIN, DIST_MAX)
    dist_gain = 1.0 / dist

    # zeitvariabler RPM-Jitter (nicht nur pro Frame)
    rpm_jitter = RPM_JITTER_BASE * np.random.randn(n_samples)
    rpm_jitter = np.convolve(rpm_jitter, np.ones(256) / 256, mode="same")  # glätten

    # mehrere AM-Modulationskomponenten
    am = np.ones_like(t)
    for rate in AM_RATES:
        phase = np.random.uniform(0, 2 * np.pi)
        depth = AM_DEPTH_BASE * np.random.uniform(0.5, 1.2)
        am += depth * np.sin(2 * np.pi * rate * t + phase)
    am = np.clip(am, 0.1, None)

    signal = np.zeros_like(t)

    # chaotische Harmoniken
    for k in range(1, N_HARM + 1):
        fk_nominal = k * f0

        # frequenzabhängige Bandbreite
        bw = HARM_BW_BASE * (1.0 + 0.5 * np.random.randn()) * (1.0 + 0.03 * k)

        # Amplitudenverhältnis mit leichter Zufallskomponente
        amp = dist_gain * (1.0 / (k ** (0.7 + 0.3 * np.random.rand())))

        # leichte spektrale Schieflage über Zeit
        fk_time = fk_nominal * (1.0 + rpm_jitter)

        # Phase zufällig
        phase0 = np.random.uniform(0, 2 * np.pi)

        # Grundsinus mit zeitvariabler Frequenz
        theta = 2 * np.pi * np.cumsum(fk_time) / SR + phase0
        base = np.sin(theta)

        # einfache "Bandbreiten"-Simulation: leichtes FM-Rauschen
        fm_noise = np.random.randn(n_samples)
        fm_noise = np.convolve(fm_noise, np.ones(64) / 64, mode="same")
        theta_bw = theta + (bw / SR) * fm_noise
        harmonic = np.sin(theta_bw)

        signal += amp * am * harmonic

    # Noise-Floor: weiß + rosa
    white = WHITE_NOISE_LEVEL * np.random.randn(n_samples)
    pink = PINK_NOISE_LEVEL * generate_pink_noise(n_samples)

    # frequenzabhängige Dämpfung (Höhen stärker gedämpft)
    # einfacher Tilt-Filter im Zeitbereich (IIR wäre besser, aber wir bleiben simpel)
    tilt = np.linspace(1.0, 0.4, n_samples)
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

    print(f"[INFO] Erzeuge {N_SAMPLES} chaotische Rotor-Sim-Drohnen")

    for i in range(N_SAMPLES):
        audio = generate_chaotic_rotor()
        sf.write(os.path.join(OUT_DIR, f"sim_drone_{i:03d}.wav"), audio, SR)

    print("[INFO] Fertig.")


if __name__ == "__main__":
    main()
