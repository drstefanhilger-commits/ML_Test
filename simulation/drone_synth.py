import numpy as np


def synthesize_drone_signal(
    sr=48000,
    duration=1.0,
    f0=120.0,
    n_harmonics=10,
    base_mag=1.0,
    decay_type="1/k2",
    decay_mag=1.0,
    roughness=0.5,
    am_depth=0.3,
    fm_depth=0.01,
    noise_level=0.02,
    jitter_amount=0.01,
    drift_amount=0.005,
):
    """
    Realistische Drohnen-Synthese:
    - Harmonische Rotorlinien
    - AM/FM-Rotor-Modulation (Sidebands)
    - Harmonische Instabilität (Jitter + Drift)
    - 1/r-Abfall + Rauhigkeit
    - Noise-Floor + Turbulenz
    """

    # ---------------------------------------------------------
    # Zeitachse
    # ---------------------------------------------------------
    N = int(sr * duration)
    t = np.arange(N) / sr

    # ---------------------------------------------------------
    # Harmonische Rotorlinien
    # ---------------------------------------------------------
    signal = np.zeros_like(t)

    # langsamer Drift (z.B. leichte Drehzahländerung)
    drift = 1.0 + drift_amount * np.sin(2 * np.pi * 0.2 * t)

    for k in range(1, n_harmonics + 1):
        fk = f0 * k

        # Amplitudenmodulation (Rotorblätter)
        am = 1.0 + am_depth * np.sin(2 * np.pi * f0 * t)

        # Frequenzmodulation (leichte Drehzahländerung)
        fm = fk * (1.0 + fm_depth * np.sin(2 * np.pi * 0.5 * t))

        # Jitter (instabile Rotorharmonische)
        jitter = 1.0 + jitter_amount * np.random.randn()

        fk_eff = fm * jitter * drift

        # harmonische Amplitude
        if decay_type == "1/k2":
            ak = base_mag / (k ** 2)
        elif decay_type == "1/k":
            ak = base_mag / k
        else:
            ak = base_mag

        signal += ak * am * np.sin(2 * np.pi * fk_eff * t)

    # ---------------------------------------------------------
    # Breitbandiger 1/r-Abfall + Rauhigkeit
    # ---------------------------------------------------------
    noise = np.random.randn(N)

    # einfacher Lowpass für realistischere Turbulenz
    alpha = 0.1
    for i in range(1, N):
        noise[i] = alpha * noise[i] + (1 - alpha) * noise[i - 1]

    signal += decay_mag * roughness * noise

    # ---------------------------------------------------------
    # Noise-Floor / Turbulenz
    # ---------------------------------------------------------
    turbulence = noise_level * np.random.randn(N)
    signal += turbulence

    # ---------------------------------------------------------
    # Normalisieren
    # ---------------------------------------------------------
    signal /= np.max(np.abs(signal) + 1e-12)

    return signal
