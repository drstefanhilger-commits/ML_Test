"""
Liste der Trainings- und Testdaten (data/48kHz) zusammenführen und dokumentieren.

Eingabe:  data/48kHz/meta_umwelt.csv, meta_dronen_real.csv, meta_dronen_sim.csv
Ausgabe:  data/48kHz/datenliste.csv       (alle Dateien, gemeinsame Spalten)
          docs/Datenliste_ML124.csv        (dieselbe Liste, versioniert im Repository)
          docs/Datenliste_ML124.md         (Übersicht + vollständige Liste nach Split/Kategorie)

Aufruf (im Repo-Wurzelverzeichnis, nach den drei Aufbereitungsskripten):
  python app/data_ml124/make_liste.py
"""
import csv, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA48, ROOT, TEST_FOLD, write_meta

COLS = ["split", "group", "subset", "category", "file", "fold", "duration_s", "source", "license"]
PARTS = ["meta_dronen_sim.csv", "meta_dronen_real.csv", "meta_umwelt.csv"]


def main():
    rows = []
    for p in PARTS:
        with open(os.path.join(DATA48, p)) as fh:
            rows += [{c: r[c] for c in COLS} for r in csv.DictReader(fh)]
    rows.sort(key=lambda r: (r["split"] != "train", r["group"], r["subset"], r["category"], r["file"]))
    for r in rows:                                     # jede Datei muss existieren
        assert os.path.isfile(os.path.join(DATA48, r["file"])), r["file"]
    write_meta(os.path.join(DATA48, "datenliste.csv"), rows)
    write_meta(os.path.join(ROOT, "docs", "Datenliste_ML124.csv"), rows)

    dur = defaultdict(float); cnt = Counter()
    for r in rows:
        k = (r["split"], r["group"], r["subset"]); cnt[k] += 1; dur[k] += float(r["duration_s"])
    L = []
    L += ["# Datenliste: Trainings- und Testdaten für Modul 124", "",
          "Automatisch erzeugt von `app/data_ml124/make_liste.py` – nicht von Hand ändern.",
          "Beschreibung der Daten, Quellen und Lizenzen: `docs/Daten_ML124.md`. Maschinenlesbar:",
          "`docs/Datenliste_ML124.csv` (identisch mit `data/48kHz/datenliste.csv`).", "",
          "> **⚠ PRÜFEN:** Die Verwendung der Drohnendaten im Training ist noch nicht freigegeben",
          "> (Prüfpunkte in `docs/Daten_ML124.md`).", "",
          "## Ablage und Aufteilung", "",
          "Alle Dateien: 48 kHz, mono (DDS: Kanal 0), WAV, unter `data/48kHz/` (nicht im Repository).",
          "Pfade in dieser Liste sind relativ zu `data/48kHz/`.", "",
          f"- **Test** = Fold {TEST_FOLD} (20 %), **Training** = Folds 1–{TEST_FOLD - 1} "
          "(die Folds bleiben für eine Kreuzvalidierung im Training erhalten).",
          "- ESC-50: Folds aus dem Datensatz, Aufteilung aber nach Originalaufnahme (`src_file`) – keine",
          "  Aufnahme liegt in Training und Test (2 Clips entgegen ihrem Fold in den Test verschoben).",
          "- DDS: Fold = (laufende Nummer − 1) % 5 + 1; synthetische Drohnen: Fold = Index % 5 + 1.",
          "- Echte Drohnen (`real_dds`) liegen **vollständig im Test**: Die Übertragbarkeit auf echte",
          "  Drohnen wird nur an echten Aufnahmen gemessen.",
          "- Beim späteren Mischen (Drohne + Umwelt) werden nur Dateien desselben Splits kombiniert.", "",
          "## Übersicht", "",
          "| Split | Gruppe | Teilmenge | Dateien | Dauer |", "|---|---|---|---|---|"]
    for k in sorted(cnt, key=lambda k: (k[0] != "train", k[1], k[2])):
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {cnt[k]} | {dur[k] / 60:.1f} min |")
    for sp in ("train", "test"):
        n = sum(v for k, v in cnt.items() if k[0] == sp); d = sum(v for k, v in dur.items() if k[0] == sp)
        L.append(f"| **{sp} gesamt** | | | **{n}** | **{d / 60:.1f} min** |")
    L += ["", "Umwelt nach Kategorie (Training / Test):", "", "| Kategorie | Training | Test |", "|---|---|---|"]
    cat = Counter((r["category"], r["split"]) for r in rows if r["group"] == "Umwelt")
    for c in sorted({r["category"] for r in rows if r["group"] == "Umwelt"}):
        L.append(f"| {c} | {cat[(c, 'train')]} | {cat[(c, 'test')]} |")

    L += ["", "## Vollständige Liste", ""]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["group"], r["subset"], r["category"])].append(r)
    last = None
    for k in sorted(groups, key=lambda k: (k[0] != "train", k[1], k[2], k[3])):
        if (k[0], k[1]) != last:
            L += ["", f"### {k[0]} / {k[1]}", ""]; last = (k[0], k[1])
        files = groups[k]
        folder = os.path.dirname(files[0]["file"])
        names = ", ".join(os.path.basename(r["file"]) + f" (F{r['fold']})" for r in files)
        L += [f"**{k[2]} / {k[3]}** – {len(files)} Dateien in `{folder}/`", "", names, ""]
    with open(os.path.join(ROOT, "docs", "Datenliste_ML124.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"{len(rows)} Einträge -> data/48kHz/datenliste.csv, docs/Datenliste_ML124.{{md,csv}}")


if __name__ == "__main__":
    main()
