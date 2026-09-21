import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# -----------------------------
# Bandpass-Filter (robust)
# -----------------------------
def bandpass_sos(audio, sr, low=80, high=600, order=4):
    nyq = 0.5 * sr
    low_n = low / nyq
    high_n = high / nyq
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfiltfilt(sos, audio)

# -----------------------------
# f0 automatisch erkennen
# -----------------------------
def detect_f0(freqs, mag, fmin=40, fmax=120):
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.argmax(mag[mask])
    return freqs[mask][idx]

# -----------------------------
# Lade reale Drohne
# -----------------------------
folder = "./data/train_48k/drone_real_train"
files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
path = os.path.join(folder, files[0])

audio, sr = sf.read(path)
if audio.ndim > 1:
    audio = audio[:,0]

# -----------------------------
# FFT vorher
# -----------------------------
n = len(audio)
window = np.hanning(n)
freqs = np.fft.rfftfreq(n, 1/sr)
fft_before = np.fft.rfft(audio * window)
mag_before = np.abs(fft_before)

# f0 vorher
f0_before = detect_f0(freqs, mag_before)

# -----------------------------
# Bandpass anwenden
# -----------------------------
audio_bp = bandpass_sos(audio, sr)

# -----------------------------
# FFT nachher
# -----------------------------
fft_after = np.fft.rfft(audio_bp * window)
mag_after = np.abs(fft_after)

# f0 nachher
f0_after = detect_f0(freqs, mag_after)

# -----------------------------
# Plot: Vorher/Nachher + Harmonische
# -----------------------------
plt.figure(figsize=(12,6))

# Vorher
plt.plot(freqs, mag_before, label="Vorher (ungefiltert)", color="gray", alpha=0.6)

# Nachher
plt.plot(freqs, mag_after, label="Nachher (Bandpass 80–1500 Hz)", color="blue")

# Harmonische markieren (nachher)
for k in range(1, 12):
    hk = k * f0_after
    if hk > 1000:
        break
    plt.axvline(hk, color="red", linestyle="--", alpha=0.5)
    plt.text(hk, np.max(mag_after)*0.8, f"{k}f0", rotation=90, color="red")

plt.title(f"FFT Vergleich – Vorher/Nachher + Harmonische (f0 ≈ {f0_after:.1f} Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
