"""Gemeinsame Hilfen für die Datenaufbereitung ML124 (siehe docs/Daten_ML124.md)."""
import os
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 48000                      # Zielrate aller Daten (Board: 47 991 Hz, Abweichung 186 ppm vernachlässigbar)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
AUDIO_ML = os.path.abspath(os.path.join(ROOT, "..", "audio_ml", "datasets"))


def to_48k_mono(path):
    """WAV lesen, Kanäle mitteln, auf 48 kHz umtasten (polyphas, Kaiser-Fenster)."""
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = x.mean(axis=1)
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
