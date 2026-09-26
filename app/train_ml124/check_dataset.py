"""
Prüfung der Beispiele aus make_dataset.py (Arbeitspaket 3): Verteilung der Band-SNR-Labels.

Kennzahlen je Misch-SNR-Klasse (Breitband):
  - positive Bänder je Frame (SNR_b > 0 dB), Median und 10/90-%-Quantil,
  - mittleres Label y_b = σ(SNR_b / 3 dB) (Trainingskonzept 5.1) = Klassenbalance,
  - Anteil Frames ohne Umweltanteil (≥ 60 Bänder mit SNR_b ≥ 70 dB, z. B. digitale Stille).
Dazu: Beispiele nur mit Umwelt (müssen 0 positive Bänder haben) und Anteil positiver Bänder je
Bandgruppe getrennt nach Training/Test.

(Der frühere Vergleich mit den Bändern der Blattharmonischen k·BPF ist entfallen: bei 2–6 Rotoren
mit unterschiedlicher Drehzahl und ±1 Band Toleranz deckt die Harmonischen-Menge fast alle Bänder ab
und trennt daher nicht.)

Ausgabe (stdout, Markdown) – Werte stehen in docs/Datensatz_ML124.md.
Aufruf (im Repo-Wurzelverzeichnis):  python app/train_ml124/check_dataset.py
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_ml124"))
from common import DATA

FEAT = os.path.join(DATA, "48kHz_features")
NB = 64
SNR_BINS = [(-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    idx = list(csv.DictReader(open(os.path.join(FEAT, "index.csv"))))
    npos = defaultdict(list); ymean = defaultdict(list); silent = defaultdict(list); n_ex = defaultdict(int)
    pos_rate = {sp: np.zeros(NB) for sp in ("train", "test")}; frames_sp = defaultdict(int)
    noise_pos = noise_vals = 0
    for r in idx:
        X = np.load(os.path.join(FEAT, r["file"]))
        cols = json.load(open(os.path.join(FEAT, r["file"][:-4] + ".json")))["columns"]
        c0 = cols.index("snr_db_00"); snr = X[:, c0:c0 + NB]
        pos = snr > 0
        pos_rate[r["split"]] += pos.sum(0); frames_sp[r["split"]] += len(X)
        if r["noise_only"] == "1":
            noise_pos += int(pos.sum()); noise_vals += pos.size
            continue
        s = float(r["snr_db"])
        k = next(f"{a}…{b} dB" for a, b in SNR_BINS if a <= s < b or (b == SNR_BINS[-1][1] and s == b))
        n_ex[k] += 1
        npos[k] += list(pos.sum(1)); ymean[k].append(float(sigmoid(snr / 3.0).mean()))
        silent[k] += list((snr >= 70).sum(1) >= 60)
    print("| Misch-SNR | Gemische | positive Bänder je Frame: Median (10 %…90 %) | mittleres Label y | Frames ohne Umwelt |")
    print("|---|---|---|---|---|")
    for a, b in SNR_BINS:
        k = f"{a}…{b} dB"
        if not n_ex[k]:
            continue
        q10, q50, q90 = np.percentile(npos[k], [10, 50, 90])
        print(f"| {k} | {n_ex[k]} | {q50:.0f} ({q10:.0f}…{q90:.0f}) | {np.mean(ymean[k]):.2f} | {np.mean(silent[k]) * 100:.1f} % |")
    allpos = np.concatenate([npos[k] for k in npos]); ally = np.concatenate([ymean[k] for k in ymean])
    print(f"\nAlle Gemische: Median {np.median(allpos):.0f} positive Bänder je Frame, mittleres Label y {ally.mean():.2f}")
    print(f"Nur Umwelt: {noise_pos} von {noise_vals} Band-Werten positiv")
    for sp in ("train", "test"):
        pr = pos_rate[sp] / frames_sp[sp] * 100
        print(f"{sp}: Anteil positiver Bänder gesamt {pr.mean():.1f} % (Band 0–15: {pr[:16].mean():.1f} %, 16–31: {pr[16:32].mean():.1f} %, "
              f"32–47: {pr[32:48].mean():.1f} %, 48–63: {pr[48:].mean():.1f} %)")


if __name__ == "__main__":
    main()
