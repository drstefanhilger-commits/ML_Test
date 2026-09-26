"""
Dronen/real_dds: 30 echte Drohnenaufnahmen aus dem Drone-detection-dataset (DDS, Svanström 2021,
CC0), 10 s, auf 48 kHz mono gewandelt (Kanal 0). Alle nach data/48kHz/test/Dronen/real_dds
(Validierung nur an echten Aufnahmen), Liste data/48kHz/meta_dronen_real.csv. Außenaufnahmen mit Hintergrund -> nur Validierung bzw.
schwache Labels (f0-Verfolgung), keine exakten Band-Labels.

⚠ PRÜFEN: Die Verwendung dieser Daten im Training ist noch nicht freigegeben und muss vorher
überprüft werden (Prüfpunkte: docs/Daten_ML124.md, Hinweis am Anfang).

Aufruf (im Repo-Wurzelverzeichnis):  python app/data_ml124/prepare_dronen_real.py
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import audio_ml, DATA48, split_of, to_48k_mono, write_meta, write_wav



def main():
    DDS = os.path.join(audio_ml(), "Drone-detection-dataset", "Data", "Audio")
    rows = []
    for f in sorted(f for f in os.listdir(DDS) if f.startswith("DRONE_") and f.endswith(".wav")):
        n = int(f.split("_")[1].split(".")[0])
        x = to_48k_mono(os.path.join(DDS, f))
        fold = (n - 1) % 5 + 1; sp = split_of(fold, kind="real")
        dst = os.path.join(sp, "Dronen", "real_dds", f)
        g = write_wav(os.path.join(DATA48, dst), x)
        rows.append(dict(file=dst, split=sp, group="Dronen", subset="real_dds", source="DDS", category="drone_real",
                         fold=fold, duration_s=round(len(x) / 48000, 3), orig=f, gain=round(g, 4), license="CC0 1.0"))
    write_meta(os.path.join(DATA48, "meta_dronen_real.csv"), rows)
    print(f"{len(rows)} Dateien nach {DATA48}/test/Dronen/real_dds")


if __name__ == "__main__":
    main()
