"""
Merkmalsdateien für die Referenzvektoren des Exports (AP5), wenn data/48kHz_features fehlt.

Reproduzierbar ohne Datensatz: zwei synthetische Test-Drohnen (Fold 5, gen_dronen_sim.synth)
mit rosa Rauschen gemischt (10 dB bzw. 0 dB) und reines rosa Rauschen, je 10 s, durch das
Merkmalswerkzeug von SDS_110 geschickt. Die Frames decken damit Drohne mit hohem/niedrigem SNR
und Rauschen ab; für den Vergleich C++ ↔ Python genügt das (die Band-Güte misst evaluate.py).

Aufruf (Repo-Wurzel, ~/ml_env):
  python app/train_ml124/make_ref_features.py <verzeichnis>
  python app/train_ml124/export.py model/ml124/k5_h48_d3 --ref <verzeichnis>/feat_*.npy
"""
import os, subprocess, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_ml124"))
import gen_dronen_sim as g
from make_dataset import tool_path

SEED_SIM = 20260926               # wie gen_dronen_sim (Seed + Index)
SEED_NOISE = 7
CASES = ((4, 10.0), (9, 0.0))     # (Drohnen-Index im Testfold, SNR dB)
NOISE_DBFS = -26.0


def pink(rng, n):
    X = np.fft.rfft(rng.standard_normal(n)); f = np.arange(len(X), dtype=float); f[0] = 1
    return np.fft.irfft(X / np.sqrt(f), n)


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def main():
    out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
    g.OUT_ROOT = out
    rng = np.random.default_rng(SEED_NOISE)
    wavs = []
    for idx, snr in CASES:
        r = g.synth(idx, SEED_SIM + idx, 10.0)
        d, sr = sf.read(os.path.join(out, r["file"]))
        nz = pink(rng, len(d)); nz *= rms(d) / rms(nz) / 10 ** (snr / 20)
        p = os.path.join(out, f"mix_{idx:04d}_{snr:.0f}dB.wav"); sf.write(p, (d + nz).astype(np.float32), sr, subtype="FLOAT")
        wavs.append(p)
    nz = pink(rng, len(d)); nz *= 10 ** (NOISE_DBFS / 20) / rms(nz)
    p = os.path.join(out, "noise_pink.wav"); sf.write(p, nz.astype(np.float32), 48000, subtype="FLOAT"); wavs.append(p)
    tool = tool_path()
    for w in wavs:
        pre = os.path.join(out, "feat_" + os.path.basename(w)[:-4])
        subprocess.run([tool, w, pre], check=True, stdout=subprocess.DEVNULL)
        print(pre + ".npy")


if __name__ == "__main__":
    main()
