import numpy as np
import librosa
import argparse
from pathlib import Path

# 1: STM32-Parameter
SR = 16000
FFT_SIZE = 256
HOP_LENGTH = 128
N_MELS = 40
N_FFT_BINS = FFT_SIZE // 2 + 1  # = 129

# 2: Mel-Filterbank laden
mel_filterbank = np.load("mel_filterbank_40x129.npy")

# 3: STM32-kompatible Mel-Features
def stm32_mel_features(y):
    window = np.hanning(FFT_SIZE)

    S = librosa.stft(
        y,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        window=window,
        center=False
    )

    mag = np.abs(S)
    mag = mag[:N_FFT_BINS, :]  # 129 Bins

    mel = mel_filterbank @ mag
    mel = np.log10(mel + 1e-6)

    feat = mel.mean(axis=1)
    return feat.astype(np.float32)

# 4: CLI
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)

# Output-Ordner deterministisch erstellen
output_dir.mkdir(parents=True, exist_ok=True)

# 5: Alle WAV-Dateien laden und verarbeiten
wav_files = sorted(input_dir.glob("*.wav"))

print(f"Gefundene WAV-Dateien: {len(wav_files)}")

for wav in wav_files:
    try:
        y, sr = librosa.load(wav, sr=SR)
        feat = stm32_mel_features(y)

        out_file = output_dir / (wav.stem + ".npy")
        np.save(out_file, feat)

        print(f"[OK] {wav.name} -> {out_file.name}")

    except Exception as e:
        print(f"[ERROR] {wav.name}: {e}")

print("Fertig.")
