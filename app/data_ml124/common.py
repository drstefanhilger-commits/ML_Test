"""Gemeinsame Hilfen für die Datenaufbereitung ML124 (siehe docs/Daten_ML124.md)."""
import os
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48000                      # Zielrate aller Daten (Board: 47 991 Hz, Abweichung 186 ppm vernachlässigbar)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
DATA48 = os.path.join(DATA, "48kHz")          # Trainings- und Testdaten, 48 kHz mono
TEST_FOLD = 5                                  # Test = Fold 5 (20 %), Training = Folds 1–4


def split_of(fold, kind=""):
    """Aufteilung Training/Test. Echte Drohnen (kind == "real") nur im Test: Übertragbarkeit auf
    echte Drohnen wird ausschließlich an echten Aufnahmen validiert (Trainingskonzept, PRÜFEN-Punkt 2)."""
    if kind == "real":
        return "test"
    return "test" if int(fold) == TEST_FOLD else "train"


def write_meta(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def _find_audio_ml():
    """Quelldatensätze (ESC-50, Drone-detection-dataset): Umgebungsvariable SDS_AUDIO_ML, sonst
    ../audio_ml/datasets neben dem Repository, sonst ../../Copilot_Projekt/audio_ml/datasets."""
    cands = [os.environ.get("SDS_AUDIO_ML", ""),
             os.path.join(ROOT, "..", "audio_ml", "datasets"),
             os.path.join(ROOT, "..", "..", "Copilot_Projekt", "audio_ml", "datasets")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "esc50")):
            return os.path.abspath(c)
    raise FileNotFoundError("Quelldatensätze nicht gefunden – SDS_AUDIO_ML auf den Ordner mit "
                            "esc50/ und Drone-detection-dataset/ setzen. Gesucht: " + ", ".join(c for c in cands if c))


AUDIO_ML = None                 # wird beim ersten Zugriff über audio_ml() aufgelöst


def audio_ml():
    global AUDIO_ML
    if AUDIO_ML is None:
        AUDIO_ML = _find_audio_ml()
    return AUDIO_ML


def to_48k_mono(path, channel=0):
    """WAV lesen, **einen** Kanal nehmen (Board: ein Referenzmikrofon; Mitteln zweier Mikrofone
    würde diffusen Schall um 1–2 dB absenken und kammfiltern), auf 48 kHz umtasten (polyphas)."""
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = x[:, min(channel, x.shape[1] - 1)]
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x


def write_wav(path, x, subtype="PCM_16"):
    """Schreibt float-Signal; skaliert nur, falls die Umtastung über Vollaussteuerung schwingt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    gain = 1.0
    if peak > 0.999:
        gain = 0.999 / peak
        x = x * gain
    sf.write(path, x, SR, subtype=subtype)
    return gain
