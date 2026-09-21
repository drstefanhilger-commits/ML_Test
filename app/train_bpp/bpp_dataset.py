import os
import numpy as np
import soundfile as sf

from app.train_bpp.bpp_bandpass import apply_bandpass
from app.train_hbo.dataset import load_dataset_single as hbo_load_single

SR = 48000
N_FFT = 2048
HOP = 512
N_HARM = 6  # wie alte 10-Feature-Pipeline


def load_wav_bandpass(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if sr != SR:
        raise ValueError(f"Erwarte SR={SR}, aber Datei hat SR={sr}")
    audio = apply_bandpass(audio, SR)
    # einfache Normierung
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio


def extract_features_from_wav(path: str) -> np.ndarray:
    audio = load_wav_bandpass(path)
    X = hbo_load_single(audio, SR, N_FFT, HOP, N_HARM)
    return X


def load_dataset_from_dir(root_dir: str) -> np.ndarray:
    X_list = []
    for f in sorted(os.listdir(root_dir)):
        if not f.lower().endswith(".wav"):
            continue
        full = os.path.join(root_dir, f)
        X = extract_features_from_wav(full)
        X_list.append(X)
    if not X_list:
        raise RuntimeError(f"Keine WAV-Dateien in {root_dir} gefunden.")
    return np.vstack(X_list)
