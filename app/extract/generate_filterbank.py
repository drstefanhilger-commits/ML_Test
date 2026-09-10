import numpy as np
import librosa

# Parameter aus deinem Mel-Extractor
sr = 16000
n_fft = 256
n_mels = 40
fmin = 0
fmax = sr / 2

mel_filterbank = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax
)

np.save("mel_filterbank_40x129.npy", mel_filterbank)
print("mel_filterbank_40x129.npy erzeugt.")
