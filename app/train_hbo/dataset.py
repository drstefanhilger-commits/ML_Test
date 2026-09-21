import os
import numpy as np
from app.train_hbo.utils_audio import load_audio
from app.train_hbo.features_harmonic import harmonic_features
from tqdm import tqdm
from simulation.dsp_core import compute_fft
from app.train_hbo.features_harmonic import harmonic_features


def load_dataset_single(audio, sr, n_fft, hop, n_harm):
    """
    Extrahiert Features aus einem einzelnen Audio-Signal.
    Identisch zur Verarbeitung in load_dataset(), aber ohne Datei-Handling.
    """
    features = harmonic_features(audio, sr, n_fft, hop, n_harm)
    return features
    

def load_dataset(folder, sr, n_fft, hop_length, n_harmonics):
    X, y = [], []

    wav_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"[INFO] Lade {len(wav_files)} Dateien aus {folder}")

    for path in tqdm(wav_files, desc=f"Processing {os.path.basename(folder)}"):
        audio = load_audio(path, sr)
        feat = harmonic_features(audio, sr, n_fft, hop_length, n_harmonics)
        X.append(feat)

        folder_name = os.path.basename(folder)
        label = 0 if "no_drone" in folder_name else 1
        y.append(label)

    return np.array(X), np.array(y)
