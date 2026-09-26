"""
data/Dronen/real_dds: 30 echte Drohnenaufnahmen aus dem Drone-detection-dataset (DDS, Svanström 2021,
CC0), 10 s, auf 48 kHz mono gewandelt. Außenaufnahmen mit Hintergrund -> nur Validierung bzw.
schwache Labels (f0-Verfolgung), keine exakten Band-Labels.

⚠ PRÜFEN: Die Verwendung dieser Daten im Training ist noch nicht freigegeben und muss vorher
überprüft werden (Prüfpunkte: docs/Daten_ML124.md, Hinweis am Anfang).

Aufruf (im Repo-Wurzelverzeichnis):  python app/data_ml124/prepare_dronen_real.py
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import AUDIO_ML, DATA, to_48k_mono, write_wav

DDS = os.path.join(AUDIO_ML, "Drone-detection-dataset", "Data", "Audio")
OUT = os.path.join(DATA, "Dronen")


def main():
    rows = []
    for f in sorted(f for f in os.listdir(DDS) if f.startswith("DRONE_") and f.endswith(".wav")):
        n = int(f.split("_")[1].split(".")[0])
        x = to_48k_mono(os.path.join(DDS, f))
        dst = os.path.join("real_dds", f)
        g = write_wav(os.path.join(OUT, dst), x)
        rows.append(dict(file=dst, source="DDS", kind="real", fold=(n - 1) % 5 + 1,
                         duration_s=round(len(x) / 48000, 3), orig=f, gain=round(g, 4), license="CC0 1.0"))
    path = os.path.join(OUT, "meta_real.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} Dateien nach {OUT}/real_dds")


if __name__ == "__main__":
    main()
