"""
Arbeitspaket 4: Auswertung eines ML124-Modells auf dem Testsplit (Fold 5) im Vergleich zum
heutigen HBD-s(t) (Spalten hbd_s_p_00…63 des Merkmalswerkzeugs, inkl. Glättung und Gate).

Kennzahlen (Trainingskonzept Abschnitt 8, „Band-Güte“):
  A) Gemischter Testsatz, Ziel „Band von der Drohne dominiert“ = SNR_b > 0 dB:
     AUC je Band (Mittel, Minimum), F1 bei θ_sel = 0,5 (gesamt), je Misch-SNR-Klasse, je Drohnentyp;
     Umwelt-Beispiele ohne Drohne: Anteil selektierter Bänder, Anteil Frames mit ≥ 1 Band.
  B) Echte Aufnahmen ohne Mischung (Hold-out, nie im Training): DDS-Drohnen (Fold 5) und alle
     Umwelt-Testclips (ESC-50, DDS-Hintergrund, DDS-Hubschrauber). Ohne Band-Labels – gezählt wird
     die Selektion: Frames mit ≥ 1 bzw. ≥ 3 Bändern, mittlere Anzahl Bänder.
ML wird wie in 124 nachbearbeitet: gleitender Mittelwert über STATE_SMOOTH_FRAMES (6) Frames je
Datei (kausal); die Werte ohne Glättung stehen daneben.

Aufruf (Repo-Wurzel, ~/ml_env):  python app/train_ml124/evaluate.py model/ml124/<name> [...]
Ausgabe: stdout (Markdown) und model/ml124/<name>/eval.json
"""
import argparse, csv, json, os, subprocess, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from ml124_data import DATA48, FEAT, NB, load_dataset, load_file, expand

THETA_SEL = 0.5
SMOOTH = 6                                   # STATE_SMOOTH_FRAMES (SDS_110_Config.hpp)
SNR_BINS = [(-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20)]
REAL = os.path.join(FEAT, "real_test")       # Merkmale der ungemischten Testaufnahmen


def smooth(p, lengths):
    """Kausaler gleitender Mittelwert über SMOOTH Frames je Datei (wie 124)."""
    out = np.empty_like(p); i = 0
    for n in lengths:
        c = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p[i:i + n]]), 0)
        k = np.arange(1, n + 1); lo = np.maximum(k - SMOOTH, 0)
        out[i:i + n] = (c[k] - c[lo]) / (k - lo)[:, None]
        i += n
    return out


def f1(yt, pp):
    tp = np.sum(yt & pp); fp = np.sum(~yt & pp); fn = np.sum(yt & ~pp)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def band_metrics(yt, p):
    aucs = [roc_auc_score(yt[:, b], p[:, b]) for b in range(NB) if 0 < yt[:, b].sum() < len(yt)]
    return dict(auc_mean=float(np.mean(aucs)), auc_min=float(np.min(aucs)), f1=float(f1(yt, p > THETA_SEL)))


def selection(p):
    s = p > THETA_SEL; n = s.sum(1)
    return dict(bands_mean=float(n.mean()), frames_ge1=float((n >= 1).mean()), frames_ge3=float((n >= 3).mean()))


# ---- ungemischte Testaufnahmen -------------------------------------------------------------
def _features(task):
    tool, wav, prefix = task
    if not os.path.isfile(prefix + ".npy"):
        r = subprocess.run([tool, "--hbd", wav, prefix], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{wav}: {r.stderr.strip()}")
    return prefix


def real_files():
    rows = list(csv.DictReader(open(os.path.join(DATA48, "datenliste.csv"))))
    out = []
    for r in rows:
        if r["split"] != "test" or r["subset"] == "sim":
            continue
        if r["group"] == "Dronen":
            grp = "Drohne (DDS)"
        elif r["subset"] == "esc50":
            grp = "ESC-50"
        else:
            grp = "DDS " + {"background": "Hintergrund", "helicopter": "Hubschrauber"}[r["category"]]
        out.append((grp, r["file"]))
    return out


def load_real(context):
    from make_dataset import tool_path
    files = real_files(); os.makedirs(REAL, exist_ok=True)
    tasks = [(tool_path(), os.path.join(DATA48, f), os.path.join(REAL, f.replace("/", "__")[:-4])) for _, f in files]
    with Pool() as pool:
        prefixes = pool.map(_features, tasks)
    data = {}
    for (grp, _), pre in zip(files, prefixes):
        d = load_file(pre + ".npy", context)
        data.setdefault(grp, []).append(d)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    cache = {}
    for mdir in a.models:
        cfg = json.load(open(os.path.join(mdir, "config.json")))
        model = tf.keras.models.load_model(os.path.join(mdir, "model.keras"))
        nz = np.load(os.path.join(mdir, "norm.npz")); k = cfg["context"]
        if k not in cache:
            cache[k] = (load_dataset("test", k), load_real(k))
        (x, snr, hbd, info, fver), real = cache[k]
        assert fver == cfg["feature_version"], (fver, cfg["feature_version"])
        predict = lambda xx: model.predict((xx - nz["mean"]) / nz["std"], batch_size=8192, verbose=0)
        lengths = [r["n"] for r in info]
        p_raw = predict(x); p = smooth(p_raw, lengths)
        yt = snr > 0
        no = expand(info, "noise_only").astype(bool); mix = ~no
        snr_mix = np.array([s if s is not None else np.nan for s in expand(info, "snr_db")], dtype=float)
        dtype = expand(info, "drone_type")
        res = dict(model=cfg["name"], feature_version=fver, test_frames=int(len(x)))
        src = dict(ML=p, ML_ungeglaettet=p_raw, HBD=hbd)
        res["gemischt"] = {n: band_metrics(yt[mix], q[mix]) for n, q in src.items()}
        res["je_snr"] = {f"{lo}…{hi} dB": {n: band_metrics(yt[m], q[m]) for n, q in src.items()}
                         for lo, hi in SNR_BINS for m in [mix & (snr_mix >= lo) & (snr_mix < hi)]}
        res["je_typ"] = {t: {n: band_metrics(yt[m], q[m]) for n, q in (("ML", p), ("HBD", hbd))}
                         for t in sorted(set(dtype[mix])) for m in [mix & (dtype == t)]}
        res["nur_umwelt"] = {n: selection(q[no]) for n, q in src.items()}
        res["echt"] = {}
        for grp, lst in real.items():
            xr = np.concatenate([d["x"] for d in lst]); pr = smooth(predict(xr), [len(d["x"]) for d in lst])
            hr = np.concatenate([d["hbd"] for d in lst])
            res["echt"][grp] = dict(dateien=len(lst), frames=int(len(xr)), ML=selection(pr), HBD=selection(hr))
        json.dump(res, open(os.path.join(mdir, "eval.json"), "w"), indent=1)
        report(res, cfg)


def report(res, cfg):
    print(f"\n## {res['model']}  (Kontext {cfg['context']}, {cfg['params']} Parameter, beste Epoche {cfg['best_epoch']}, "
          f"val_loss {cfg['val_loss']:.4f}{', ohne %d Rotoren trainiert' % cfg['exclude_rotors'] if cfg['exclude_rotors'] else ''})\n")
    print("| Teilmenge | ML AUC Mittel / Min | ML F1 | ML ungeglättet AUC / F1 | HBD AUC Mittel / Min | HBD F1 |")
    print("|---|---|---|---|---|---|")
    rows = [("alle Gemische", res["gemischt"])] + list(res["je_snr"].items())
    for name, m in rows:
        print(f"| {name} | {m['ML']['auc_mean']:.3f} / {m['ML']['auc_min']:.3f} | {m['ML']['f1']:.3f} | "
              f"{m['ML_ungeglaettet']['auc_mean']:.3f} / {m['ML_ungeglaettet']['f1']:.3f} | "
              f"{m['HBD']['auc_mean']:.3f} / {m['HBD']['auc_min']:.3f} | {m['HBD']['f1']:.3f} |")
    print("\n| Drohnentyp (Rotoren×Blätter) | ML AUC | ML F1 | HBD AUC | HBD F1 |\n|---|---|---|---|---|")
    for t, m in res["je_typ"].items():
        print(f"| {t} | {m['ML']['auc_mean']:.3f} | {m['ML']['f1']:.3f} | {m['HBD']['auc_mean']:.3f} | {m['HBD']['f1']:.3f} |")
    print("\n| Aufnahmen | Dateien | ML: Frames ≥1 / ≥3 Bänder, Bänder im Mittel | HBD: Frames ≥1 / ≥3 Bänder, Bänder im Mittel |\n|---|---|---|---|")
    fmt = lambda s: f"{s['frames_ge1'] * 100:.1f} % / {s['frames_ge3'] * 100:.1f} %, {s['bands_mean']:.2f}"
    no = res["nur_umwelt"]
    print(f"| Testsatz nur Umwelt (gemischt erzeugt) | – | {fmt(no['ML'])} | {fmt(no['HBD'])} |")
    for grp, m in sorted(res["echt"].items()):
        print(f"| {grp} (echt, ungemischt) | {m['dateien']} | {fmt(m['ML'])} | {fmt(m['HBD'])} |")


if __name__ == "__main__":
    main()
