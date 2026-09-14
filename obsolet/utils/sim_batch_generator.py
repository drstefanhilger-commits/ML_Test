import numpy as np
from pathlib import Path
from app.utils.sim_exporter import export_simulation_wav

# Zielordner für Batch-Simulationen
BATCH_DIR = Path("data/train/drone_sim")
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def generate_simulation_batch(
    angles=None,
    distances=None,
    rpms=None,
    noise_levels=None,
    wind_levels=None,
    sds_gain=1.0,
    prefix="sim"
):
    """
    Erzeugt einen kompletten Batch realistischer Drohnensimulationen.
    Jede Kombination aus:
        - angle
        - distance
        - rpm
        - noise_level
        - wind_level
    wird als eigene WAV-Datei exportiert.
    """

    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

    if distances is None:
        distances = [20, 30, 40, 50, 60, 70]

    if rpms is None:
        rpms = [1600, 1800, 2000, 2200]

    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15]

    if wind_levels is None:
        wind_levels = [0.02, 0.04, 0.06]

    count = 0

    for angle in angles:
        for dist in distances:
            for rpm in rpms:
                for nl in noise_levels:
                    for wl in wind_levels:

                        filename = (
                            f"{prefix}_A{angle}_D{dist}_RPM{rpm}_N{nl}_W{wl}.wav"
                        )

                        export_simulation_wav(
                            angle_deg=angle,
                            distance_m=dist,
                            rpm=rpm,
                            noise_level=nl,
                            wind_level=wl,
                            sds_gain=sds_gain,
                            filename=filename
                        )

                        count += 1

    print(f"\n[OK] Batch-Simulation abgeschlossen: {count} Dateien erzeugt.")
    return count
