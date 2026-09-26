"""
data/Umwelt: ESC-50 (2000 Clips, 50 Klassen) und Drone-detection-dataset (DDS, Svanström 2021):
BACKGROUND und HELICOPTER. Ausgabe 48 kHz mono, meta.csv mit Quelle, Kategorie, Fold, Lizenz.

Folds: ESC-50 bringt 5 Folds mit (Clips derselben Originalaufnahme liegen im selben Fold).
DDS: Fold = (laufende Nummer − 1) % 5 + 1 (Aufnahmezusammenhang unbekannt).

Aufruf (im Repo-Wurzelverzeichnis):  python app/data_ml124/prepare_umwelt.py
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import AUDIO_ML, DATA, to_48k_mono, write_wav

ESC = os.path.join(AUDIO_ML, "esc50", "ESC-50-master")
DDS = os.path.join(AUDIO_ML, "Drone-detection-dataset", "Data", "Audio")
OUT = os.path.join(DATA, "Umwelt")


def main():
    rows = []
    with open(os.path.join(ESC, "meta", "esc50.csv")) as f:
        for r in csv.DictReader(f):
            dst = os.path.join("esc50", r["category"], r["filename"])
            g = write_wav(os.path.join(OUT, dst), to_48k_mono(os.path.join(ESC, "audio", r["filename"])))
            rows.append(dict(file=dst, source="ESC-50", category=r["category"], fold=r["fold"], duration_s=5.0,
                             orig=r["filename"], gain=round(g, 4), license="CC BY-NC 3.0"))
    for cat in ("BACKGROUND", "HELICOPTER"):
        files = sorted(f for f in os.listdir(DDS) if f.startswith(cat + "_") and f.endswith(".wav"))
        for f in files:
            n = int(f.split("_")[1].split(".")[0])
            x = to_48k_mono(os.path.join(DDS, f))
            dst = os.path.join("dds", cat.lower(), f)
            g = write_wav(os.path.join(OUT, dst), x)
            rows.append(dict(file=dst, source="DDS", category=cat.lower(), fold=(n - 1) % 5 + 1,
                             duration_s=round(len(x) / 48000, 3), orig=f, gain=round(g, 4), license="CC0 1.0"))
    with open(os.path.join(OUT, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} Dateien nach {OUT}")


if __name__ == "__main__":
    main()
