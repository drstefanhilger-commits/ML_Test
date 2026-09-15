#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BUILD FEATURES V7 — Pfad‑B, deterministisch

Dieses Skript erzeugt Mel‑Spectrogram‑Features aus den V7‑WAV-Dateien.

NEU:
----
Vor dem Erzeugen der Features werden ALLE alten .npy-Dateien in
data/features_v7/... automatisch gelöscht, damit keine Alt‑Influenz
oder inkonsistente Feature‑Längen bestehen bleiben.

Warum fixe Länge?
-----------------
WAV-Dateien haben unterschiedliche Dauer → unterschiedliche Anzahl Mel‑Frames.
Flatten() erzeugt dann Feature‑Vektoren unterschiedlicher Länge → Training bricht ab.

Lösung:
-------
Wir erzwingen FIXED_FRAMES = 256 Mel‑Frames.
- Kürzer → rechts mit Nullen auffüllen
- Länger → abschneiden

Feature‑Vektor-Länge:
256 Frames × 40 Mel‑Bins = 10240 Werte
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil

# ---------------------------------------------------------
# WAV‑Ordner (V7)
# ---------------------------------------------------------
WAV_DIRS = {
    "drone_real": Path("data/train_48k/drone_real_train"),
    "no_drone":   Path("data/train_48k/no_drone_train"),
}

# ---------------------------------------------------------
# Feature‑Ordner (V7)
# ---------------------------------------------------------
FEATURE_DIRS = {
    "drone_real": Path("data/features_v7/drone_real"),
    "no_drone":   Path("data/features_v7/no_drone"),
}

# ---------------------------------------------------------
# DSP‑Parameter (müssen zu check_v7.py passen)
# ---------------------------------------------------------
SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 40
FMIN = 20
FMAX = 20000

# ---------------------------------------------------------
# FIXE Anzahl Mel‑Frames
# ---------------------------------------------------------
FIXED_FRAMES = 256   # garantiert kompatibel mit train_v7.py

# ---------------------------------------------------------
# Alte Features löschen (Pfad‑B)
# ---------------------------------------------------------
def delete_old_features():
    print("======================================")
    print("      DELETE OLD FEATURES V7")
    print("======================================")

    for cls, d in FEATURE_DIRS.items():
        if d.exists():
            print(f"[DEL] Lösche Ordner: {d}")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("======================================\n")

# ---------------------------------------------------------
# Feature‑Extraktion
# ---------------------------------------------------------
def extract_features(wav_path: Path):
    """
    Erzeugt ein Mel‑Spectrogram und erzwingt eine fixe Frame‑Länge.
    """

    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        print(f"[ERR] {wav_path.name}: {e}")
        return None

    if sr != SR:
        print(f"[WARN] {wav_path.name}: Sample‑Rate {sr}, erwartet 48000")
        return None

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    frames = mel_db.shape[1]

    # Kürzer → rechts auffüllen
    if frames < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")

    # Länger → abschneiden
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    feat = mel_db.flatten().astype(np.float32)
    return feat

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("======================================")
    print("          BUILD FEATURES V7")
    print("======================================")

    # 1) Alte Features löschen
    delete_old_features()

    # 2) Neue Features erzeugen
    total = 0

    for cls, wav_dir in WAV_DIRS.items():
        out_dir = FEATURE_DIRS[cls]

        print(f"\n[INFO] Klasse: {cls}")
        print(f"[INFO] WAV‑Ordner: {wav_dir}")

        wavs = sorted(wav_dir.glob("*.wav"))
        print(f"[INFO] Anzahl WAVs: {len(wavs)}")

        for wav in wavs:
            feat = extract_features(wav)
            if feat is None:
                print(f"[ERROR] {wav.name}")
                continue

            out_path = out_dir / (wav.stem + ".npy")
            np.save(out_path, feat)
            # print(f"[OK] {out_path.name}")

            total += 1

    print("\n======================================")
    print(f"  BUILD FEATURES V7 DONE — {total} Dateien")
    print("======================================")


if __name__ == "__main__":
    main()
