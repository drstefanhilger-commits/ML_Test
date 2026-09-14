import numpy as np

def generate_realistic_drone_signal(
    frame_len=256,
    sample_rate=16000,
    rpm=1800,
    num_blades=2,
    harmonics=6,
    noise_level=0.15,
    wind_level=0.05
):
    """
    Erzeugt ein realistisches Drohnensignal:
    - Rotorharmonische
    - Blade-Pass-Frequenz (BPF)
    - RPM-Modulation
    - Motorrauschen
    - Windmaskierung
    """

    t = np.arange(frame_len) / sample_rate

    # Rotor-Grundfrequenz (Hz)
    f0 = rpm / 60.0

    # Blade-Pass-Frequenz
    bpf = f0 * num_blades

    # Harmonische Struktur
    sig = np.zeros(frame_len, dtype=np.float32)
    for h in range(1, harmonics + 1):
        amp = 1.0 / h
        sig += amp * np.sin(2 * np.pi * h * f0 * t)

    # Blade-Pass-Komponente
    sig += 0.8 * np.sin(2 * np.pi * bpf * t)

    # leichte RPM-Modulation (±5%)
    mod = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    sig *= mod

    # Motorrauschen
    sig += noise_level * np.random.randn(frame_len)

    # Windmaskierung (breitbandig)
    wind = wind_level * np.random.randn(frame_len)
    sig += wind

    # Normalisieren
    sig /= np.max(np.abs(sig) + 1e-6)

    return sig.astype(np.float32)
