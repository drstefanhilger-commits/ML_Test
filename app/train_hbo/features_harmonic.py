import numpy as np
import librosa
from app.train_hbo.f0_estimator import estimate_f0_v3

def harmonic_features(y, sr, n_fft, hop_length, n_harmonics):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Alte F0-Schätzung: immer 6 Harmonische
    f0 = estimate_f0_v3(
        S, freqs,
        rotor_min=80.0,
        rotor_max=200.0,
        n_harmonics=6,
        freq_tol=5.0
    )

    # Alte Harmonische: exakt 6
    harmonics = [f0 * n for n in range(1, 7)]

    harmonic_energy = []
    for h in harmonics:
        idx = np.argmin(np.abs(freqs - h))
        harmonic_energy.append(np.mean(S[idx, :]))

    total_energy = np.sum(S)
    harmonic_ratio = np.sum(harmonic_energy) / (total_energy + 1e-9)
    harmonic_spread = np.std(harmonic_energy)
    harmonic_stability = np.mean(np.diff(harmonic_energy))

    # Exakt 10 Features → alter Zustand
    return np.array([
        f0,
        harmonic_ratio,
        harmonic_spread,
        harmonic_stability,
        *harmonic_energy
    ])
