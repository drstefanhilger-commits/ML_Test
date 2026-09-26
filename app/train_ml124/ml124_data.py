"""
Gemeinsame Lade- und Aufbereitungsfunktionen für Training und Auswertung von Modul 124 (AP4).

Eingang je Frame: die 169 Merkmale von 122 (band_log_power, mel, spectral_flux, band_am_depth),
bei Kontext K die Merkmale der Frames t, t−1 … t−K+1 hintereinander (neuester zuerst) – so wird
124 sie auf dem Board stapeln. Ziel: y_b = σ(SNR_b / 3 dB) (Trainingskonzept 5.1).
Die ersten SKIP_FRAMES Frames jeder Datei (Einschwingen von AGC, Rauschboden, AM-Historie, HBD)
werden verworfen (docs/Datensatz_ML124.md, Abschnitt 6).
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_ml124"))
from common import DATA, DATA48, ROOT

FEAT = os.path.join(DATA, "48kHz_features")
NB = 64
N_FEAT = 169
SKIP_FRAMES = 31                  # 1 s
LABEL_SCALE_DB = 3.0


def feature_columns(cols):
    """Spaltenindizes der 169 Merkmale in der Reihenfolge von 122."""
    i0 = cols.index("band_log_power_00")
    idx = list(range(i0, i0 + N_FEAT))
    assert cols[idx[-1]] == "band_am_depth_63", cols[idx[-1]]
    return idx


def stack_context(F, k):
    """[T, 169] -> [T, 169·k]; Zeile t = (F[t], F[t−1], …); am Anfang mit Frame 0 aufgefüllt."""
    parts = [F]
    for d in range(1, k):
        parts.append(np.concatenate([np.repeat(F[:1], d, 0), F[:-d]], 0))
    return np.concatenate(parts, 1)


def load_file(path, context):
    """Eine .npy/.json des Merkmalswerkzeugs -> dict mit x, snr (falls vorhanden), hbd s(t)."""
    X = np.load(path); cols = json.load(open(path[:-4] + ".json"))["columns"]
    F = X[:, feature_columns(cols)]
    out = dict(x=stack_context(F, context)[SKIP_FRAMES:], feature_version=json.load(open(path[:-4] + ".json"))["feature_version"])
    if "snr_db_00" in cols:
        c = cols.index("snr_db_00"); out["snr"] = X[SKIP_FRAMES:, c:c + NB]
    if "hbd_s_p_00" in cols:
        c = cols.index("hbd_s_p_00"); out["hbd"] = X[SKIP_FRAMES:, c:c + NB]
    return out


def drone_meta():
    with open(os.path.join(DATA48, "meta_dronen_sim.csv")) as fh:
        return {r["file"]: r for r in csv.DictReader(fh)}


def load_dataset(split, context):
    """Alle Beispiele eines Splits aus data/48kHz_features. Rückgabe: x, snr, hbd, Frame-Infos."""
    idx = [r for r in csv.DictReader(open(os.path.join(FEAT, "index.csv"))) if r["split"] == split]
    dm = drone_meta()
    xs, snrs, hbds, info = [], [], [], []
    versions = set()
    for r in idx:
        d = load_file(os.path.join(FEAT, r["file"]), context)
        versions.add(d["feature_version"])
        n = len(d["x"]); xs.append(d["x"]); snrs.append(d["snr"]); hbds.append(d["hbd"])
        m = dm.get(r["drone_file"], {})
        info.append(dict(id=r["id"], n=n, noise_only=r["noise_only"] == "1", drone_file=r["drone_file"],
                         snr_db=float(r["snr_db"]) if r["snr_db"] else None,
                         drone_type=f"{m['n_rotors']}x{m['n_blades']}" if m else ""))
    assert len(versions) == 1, versions
    return (np.concatenate(xs).astype(np.float32), np.concatenate(snrs).astype(np.float32),
            np.concatenate(hbds).astype(np.float32), info, versions.pop())


def expand(info, key):
    """Frame-weiser Vektor eines Beispiel-Attributs."""
    return np.concatenate([np.full(r["n"], r[key], dtype=object) for r in info])


def soft_target(snr):
    return (1.0 / (1.0 + np.exp(-snr / LABEL_SCALE_DB))).astype(np.float32)


def git_commit(path=ROOT):
    try:
        import subprocess
        return subprocess.run(["git", "-C", path, "describe", "--always", "--dirty"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""
