import numpy as np

# ----------------------------------------
# CONFIG
# ----------------------------------------
SDS_SAMPLE_RATE = 16000
SDS_FRAME_LEN = 4096

# ----------------------------------------
# Realistic synthetic drone source
# ----------------------------------------
def generate_sds_drone_source(
        f0=180.0,            # Grundfrequenz (Rotor)
        harmonics=[1,2,3,4], # Fourier-Harmonische
        harmonic_gain=[1.0, 0.6, 0.35, 0.2],
        amp_mod_freq=12.0,   # langsame Modulation (Rotor-Flattern)
        amp_mod_depth=0.3,
        freq_mod_freq=3.0,   # leichte Frequenzmodulation
        freq_mod_depth=0.02,
        noise_level=0.05     # breitbandiges Rauschen
    ):
    """
    Erzeugt ein synthetisches Drohnensignal mit:
    - mehreren Harmonischen
    - Amplitudenmodulation
    - Frequenzmodulation
    - breitbandigem Rauschen
    """

    t = np.arange(SDS_FRAME_LEN) / SDS_SAMPLE_RATE

    # Frequenzmodulation (FM)
    fm = freq_mod_depth * np.sin(2 * np.pi * freq_mod_freq * t)
    f_inst = f0 * (1.0 + fm)

    # Grundsignal
    sig = np.zeros_like(t, dtype=np.float32)

    # Harmonische addieren
    for k, g in zip(harmonics, harmonic_gain):
        sig += g * np.sin(2 * np.pi * f_inst * k * t)

    # Amplitudenmodulation (AM)
    am = 1.0 + amp_mod_depth * np.sin(2 * np.pi * amp_mod_freq * t)
    sig *= am

    # Breitbandiges Rauschen
    sig += noise_level * np.random.randn(len(t))

    # leichte Normalisierung
    sig /= np.max(np.abs(sig) + 1e-6)

    return sig.astype(np.float32)

# ----------------------------------------
# Test
# ----------------------------------------
if __name__ == "__main__":
    src = generate_sds_drone_source()
    print("SDS-Drone-Quelle erzeugt. Länge:", len(src))
