import numpy as np
import soundfile as sf
from pathlib import Path

# KORREKTE Imports für deine Projektstruktur
from app.utils.drone_signal_realistic import generate_realistic_drone_signal
from app.utils.sds_simulation import make_sds_frame

SAMPLE_RATE = 16000
FRAME_LEN = 256

# Export-Ziel
EXPORT_DIR = Path("data/train/drone_sim")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_simulation_wav(
    angle_deg=0,
    distance_m=30,
    rpm=1800,
    noise_level=0.12,
    wind_level=0.04,
    sds_gain=1.0,
    filename=None
):
    """
    Erzeugt eine realistische Drohnensimulation und speichert sie als WAV.
    - Rotorharmonische
    - Blade-Pass-Frequenz
    - RPM-Modulation
    - Motorrauschen
    - Windmaskierung
    - SDS-Delay-Geometrie
    """

    # 1) Realistisches Drohnensignal erzeugen
    drone_src = generate_realistic_drone_signal(
        frame_len=FRAME_LEN,
        sample_rate=SAMPLE_RATE,
        rpm=rpm,
        num_blades=2,
        harmonics=6,
        noise_level=noise_level,
        wind_level=wind_level
    )

    # 2) SDS-Frame erzeugen (Delay-Geometrie)
    sds_frame = make_sds_frame(angle_deg, distance_m, noise_level)

    # 3) Mono-SDS erzeugen
    sds_mono = sds_frame.mean(axis=0)

    # 4) Mischung
    mix = drone_src + sds_gain * sds_mono

    # 5) Normalisieren
    mix /= np.max(np.abs(mix) + 1e-6)

    # 6) Dateiname erzeugen
    if filename is None:
        filename = f"sim_A{angle_deg}_D{distance_m}_RPM{rpm}.wav"

    out_path = EXPORT_DIR / filename

    # 7) WAV speichern
    sf.write(out_path, mix.astype(np.float32), SAMPLE_RATE)

    print(f"[OK] Simulation gespeichert: {out_path}")
    return out_path
