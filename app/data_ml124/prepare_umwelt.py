"""
Umwelt: ESC-50 (2000 Clips, 50 Klassen) und Drone-detection-dataset (DDS, Svanström 2021):
BACKGROUND und HELICOPTER. Ausgabe 48 kHz mono (DDS: Kanal 0) nach
data/48kHz/{train,test}/Umwelt/..., Liste data/48kHz/meta_umwelt.csv (Quelle, Kategorie, Fold, Split, Lizenz).

Folds: ESC-50 bringt 5 Folds mit. Laut Datensatz liegen Clips derselben Originalaufnahme im selben
Fold – 4 Aufnahmen verletzen das, 2 davon über die Grenze Training/Test (src_file 209698, 234879).
Deshalb wird nach Originalaufnahme (src_file) aufgeteilt: liegt ein Clip im Test-Fold, kommen alle
Clips dieser Aufnahme in den Test (betrifft 4-209698-A-37.wav, 4-234879-A-6.wav).
DDS: Fold = (laufende Nummer − 1) % 5 + 1 (Aufnahmezusammenhang unbekannt).

Aufruf (im Repo-Wurzelverzeichnis):  python app/data_ml124/prepare_umwelt.py
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import AUDIO_ML, DATA48, split_of, to_48k_mono, write_meta, write_wav

ESC = os.path.join(AUDIO_ML, "esc50", "ESC-50-master")
DDS = os.path.join(AUDIO_ML, "Drone-detection-dataset", "Data", "Audio")


def main():
    rows = []
    with open(os.path.join(ESC, "meta", "esc50.csv")) as f:
        esc = list(csv.DictReader(f))
    test_src = {r["src_file"] for r in esc if split_of(r["fold"]) == "test"}   # Aufteilung nach Originalaufnahme
    for r in esc:
        sp = "test" if r["src_file"] in test_src else split_of(r["fold"])
        dst = os.path.join(sp, "Umwelt", "esc50", r["category"], r["filename"])
        g = write_wav(os.path.join(DATA48, dst), to_48k_mono(os.path.join(ESC, "audio", r["filename"])))
        rows.append(dict(file=dst, split=sp, group="Umwelt", subset="esc50", source="ESC-50", category=r["category"],
                         fold=r["fold"], duration_s=5.0, orig=r["filename"], gain=round(g, 4), license="CC BY-NC 3.0"))
    for cat in ("BACKGROUND", "HELICOPTER"):
        files = sorted(f for f in os.listdir(DDS) if f.startswith(cat + "_") and f.endswith(".wav"))
        for f in files:
            n = int(f.split("_")[1].split(".")[0])
            x = to_48k_mono(os.path.join(DDS, f))
            fold = (n - 1) % 5 + 1; sp = split_of(fold)
            dst = os.path.join(sp, "Umwelt", "dds", cat.lower(), f)
            g = write_wav(os.path.join(DATA48, dst), x)
            rows.append(dict(file=dst, split=sp, group="Umwelt", subset="dds", source="DDS", category=cat.lower(),
                             fold=fold, duration_s=round(len(x) / 48000, 3), orig=f, gain=round(g, 4), license="CC0 1.0"))
    write_meta(os.path.join(DATA48, "meta_umwelt.csv"), rows)
    print(f"{len(rows)} Dateien nach {DATA48}/{{train,test}}/Umwelt")


if __name__ == "__main__":
    main()
