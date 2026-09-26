"""
Arbeitspaket 3 (docs/Trainingskonzept_ML124.md, Abschnitt 5): Trainings- und Testbeispiele für
Modul 124 – Drohne + Umwelt mit kontrolliertem SNR mischen, Merkmale und Band-SNR mit dem
Merkmalswerkzeug von SDS_110 berechnen (`sds_features --hbd --label`, Board-Code).

⚠ PRÜFEN: Die Verwendung der Drohnendaten im Training ist noch nicht freigegeben
(docs/Daten_ML124.md). Die erzeugten Beispiele erben diesen Vorbehalt.

Je Beispiel:
  - Drohnenanteil: synthetische Drohne (Dronen/sim) desselben Splits, 10 s; bei „nur Umwelt“ 0.
  - Umweltanteil: 10-s-Spur aus zufälligen Umwelt-Clips desselben Splits (ESC-50, DDS Hintergrund/
    Hubschrauber), je Clip auf gleichen RMS normiert, mit 20 ms Überblendung aneinandergehängt.
    Darunter liegt stets ein Umgebungs-Grundpegel (DDS-Hintergrund desselben Splits, 10–25 dB unter
    der Spur): Draußen gibt es keine digitale Stille, ESC-50 enthält sie aber (aufgefüllte Clips).
  - SNR = 10·log10(P(Drohne) / P(Umwelt)) über den ganzen Ausschnitt (Breitband), gleichverteilt.
  - Pegel des Gemischs (RMS) gleichverteilt; Spitze auf höchstens −1 dBFS begrenzt (sonst Pegel gesenkt).
  - Beide Anteile als float32-WAV in einen temporären Ordner, dann sds_features; WAVs werden gelöscht.

Ausgabe: data/48kHz_features/{train,test}/<id>.npy/.json (Spalten siehe JSON: t_s, 169 Merkmale,
69 HBD-Spalten, snr_db_00…63), data/48kHz_features/index.csv (Rezept und Kennzahlen je Beispiel).
Labels: y_b = σ(SNR_b / 3 dB) (Trainingskonzept 5.1) werden erst im Training gebildet.

Aufruf (im Repo-Wurzelverzeichnis):
  python app/train_ml124/make_dataset.py [--jobs 16] [--limit N]
Werkzeug: Umgebungsvariable SDS_FEATURES, sonst ../SDS_110/build/tools/sds_features
(bauen mit: make -C ../SDS_110/tools/features).
"""
import argparse, csv, json, os, shutil, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_ml124"))
from common import DATA, DATA48, ROOT, SR, write_meta

OUT = os.path.join(DATA, "48kHz_features")

# ---- Parameter (Trainingskonzept 5.1/5.2) -------------------------------------------------
SEED = 20260927
SEG_S = 10.0                      # Länge eines Beispiels
MIX_PER_DRONE = 4                 # Gemische je Drohnen-Clip
NOISE_ONLY_FRACTION = 0.25        # zusätzliche Beispiele nur mit Umwelt (Anteil an den Gemischen)
SNR_DB = (-30.0, 20.0)            # Breitband-SNR Drohne/Umwelt (−10…+30 dB ergab 71 % positive Bänder)
AMBIENT_DB = (-25.0, -10.0)       # Umgebungs-Grundpegel relativ zur Umwelt-Spur
LEVEL_DBFS = (-45.0, -15.0)       # RMS des Gemischs
PEAK_MAX_DBFS = -1.0
XFADE_S = 0.02


def tool_path():
    p = os.environ.get("SDS_FEATURES") or os.path.join(ROOT, "..", "SDS_110", "build", "tools", "sds_features")
    if not os.path.isfile(p):
        sys.exit(f"Merkmalswerkzeug nicht gefunden: {p} (make -C ../SDS_110/tools/features oder SDS_FEATURES setzen)")
    return os.path.abspath(p)


def load_list():
    with open(os.path.join(DATA48, "datenliste.csv")) as fh:
        rows = list(csv.DictReader(fh))
    pools = {}
    for sp in ("train", "test"):
        pools[sp] = dict(drone=[r["file"] for r in rows if r["split"] == sp and r["subset"] == "sim"],
                         noise=[r["file"] for r in rows if r["split"] == sp and r["group"] == "Umwelt"],
                         ambient=[r["file"] for r in rows if r["split"] == sp and r["category"] == "background"])
    return pools


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def noise_track(rng, files, n):
    """10-s-Spur aus zufälligen Clips, je Clip auf RMS 1 normiert (stille Clips übersprungen)."""
    out = np.zeros(0); used = []; xf = int(XFADE_S * SR)
    while len(out) < n:
        f = files[rng.integers(len(files))]
        x, sr = sf.read(os.path.join(DATA48, f), dtype="float64")
        assert sr == SR
        r = rms(x)
        if r < 1e-6:
            continue
        x = x / r; used.append(f)
        if len(out) == 0:
            out = x
        else:
            w = np.linspace(0, 1, xf)
            out = np.concatenate([out[:-xf], out[-xf:] * (1 - w) + x[:xf] * w, x[xf:]])
    return out[:n], used


def build_components(split, drone_file, noise_only, seed, pools):
    """Anteile eines Beispiels (reproduzierbar aus dem Seed) -> Drohne, Umwelt (float64), Rezept-Teile."""
    rng = np.random.default_rng(seed)
    n = int(SEG_S * SR)
    noise, used = noise_track(rng, pools[split]["noise"], n)
    amb, amb_used = noise_track(rng, pools[split]["ambient"], n)
    amb_db = float(rng.uniform(*AMBIENT_DB))
    noise = noise + amb * 10 ** (amb_db / 20)
    noise = noise / rms(noise)
    if noise_only:
        drone = np.zeros(n); snr = None
    else:
        drone, sr = sf.read(os.path.join(DATA48, drone_file), dtype="float64"); assert sr == SR
        drone = drone[:n] / rms(drone[:n])
        snr = float(rng.uniform(*SNR_DB))
        noise = noise * 10 ** (-snr / 20)                      # P(Drohne)/P(Umwelt) = SNR
    mix_rms = rms(drone + noise)
    level = float(rng.uniform(*LEVEL_DBFS))
    g = 10 ** (level / 20) / mix_rms
    peak = float(np.max(np.abs(drone + noise))) * g
    if peak > 10 ** (PEAK_MAX_DBFS / 20):                      # Spitze begrenzen -> Pegel senken
        g *= 10 ** (PEAK_MAX_DBFS / 20) / peak
        level = 20 * np.log10(g * mix_rms)
    d, nz = (drone * g).astype(np.float32), (noise * g).astype(np.float32)
    return d, nz, dict(noise_files=";".join(used), ambient_files=";".join(amb_used), ambient_db=round(amb_db, 2),
                       snr_db="" if snr is None else round(snr, 2), level_dbfs=round(float(level), 2))


def make_example(task):
    split, idx, drone_file, noise_only, seed, tool, tmp_root = task
    d, nz, parts = build_components(split, drone_file, noise_only, seed, POOLS)
    ex = f"{split}_{idx:05d}"
    tmp = tempfile.mkdtemp(prefix=ex + "_", dir=tmp_root)
    try:
        fd, fn = os.path.join(tmp, "d.wav"), os.path.join(tmp, "n.wav")
        sf.write(fd, d, SR, subtype="FLOAT"); sf.write(fn, nz, SR, subtype="FLOAT")
        out_prefix = os.path.join(OUT, split, ex)
        r = subprocess.run([tool, "--hbd", "--label", fd, fn, out_prefix], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{ex}: sds_features {r.returncode}: {r.stderr.strip()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    meta = json.load(open(out_prefix + ".json"))
    recipe = dict(id=ex, split=split, noise_only=int(noise_only), drone_file=drone_file or "", seed=seed, **parts)
    meta["recipe"] = recipe
    json.dump(meta, open(out_prefix + ".json", "w"), indent=1)
    return dict(recipe, file=f"{split}/{ex}.npy", frames=meta["frames"], clipped=meta["clipped_samples"],
                max_rec_err=meta["max_reconstruction_error"], feature_version=meta["feature_version"])


def _init(pools):
    global POOLS
    POOLS = pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Beispiele je Split (Probelauf)")
    a = ap.parse_args()
    tool = tool_path(); pools = load_list()
    tmp_root = tempfile.mkdtemp(prefix="ml124_mix_")
    tasks = []
    for s_i, sp in enumerate(("train", "test")):
        os.makedirs(os.path.join(OUT, sp), exist_ok=True)
        drones = pools[sp]["drone"]
        jobs = [(f, False) for f in drones for _ in range(MIX_PER_DRONE)]
        jobs += [(None, True)] * int(round(len(jobs) * NOISE_ONLY_FRACTION))
        order = np.random.default_rng(SEED + s_i).permutation(len(jobs))
        jobs = [jobs[i] for i in order]
        if a.limit:
            jobs = jobs[:a.limit]
        tasks += [(sp, i, f, no, SEED + 1_000_000 * (s_i + 1) + i, tool, tmp_root) for i, (f, no) in enumerate(jobs)]
    try:
        with Pool(a.jobs, initializer=_init, initargs=(pools,)) as pool:
            rows = pool.map(make_example, tasks, chunksize=4)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    write_meta(os.path.join(OUT, "index.csv"), rows)
    print(f"{len(rows)} Beispiele nach {OUT} (Training {sum(r['split'] == 'train' for r in rows)}, "
          f"Test {sum(r['split'] == 'test' for r in rows)})")


if __name__ == "__main__":
    main()
