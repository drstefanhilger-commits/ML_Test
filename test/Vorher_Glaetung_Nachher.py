import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, savgol_filter

# -----------------------------
# Bandpass-Filter
# -----------------------------
def bandpass_sos(audio, sr, low=80, high=600, order=4):
    nyq = 0.5 * sr
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
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
folder = "./data/train_48k/drone_real_test"
# folder = "./data/train_48k/drone_sim_test"
files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
path = os.path.join(folder, files[0])

audio, sr = sf.read(path)
if audio.ndim > 1:
    audio = audio[:,0]

# -----------------------------
# Bandpass
# -----------------------------
audio_bp = bandpass_sos(audio, sr)

# -----------------------------
# FFT roh
# -----------------------------
n = len(audio_bp)
window = np.hanning(n)
freqs = np.fft.rfftfreq(n, 1/sr)
fft_mag = np.abs(np.fft.rfft(audio_bp * window))

# -----------------------------
# FFT glätten (Savitzky-Golay)
# -----------------------------
fft_smooth = savgol_filter(fft_mag, window_length=101, polyorder=3)

# -----------------------------
# f0 erkennen (roh)
# -----------------------------
f0 = detect_f0(freqs, fft_mag)

# -----------------------------
# Plot: FFT roh + FFT geglättet + Harmonische
# -----------------------------
plt.figure(figsize=(12,6))

# Rohes Spektrum
plt.plot(freqs, fft_mag, label="FFT Bandpass (roh)", color="gray", alpha=0.5)

# Geglättetes Spektrum
plt.plot(freqs, fft_smooth, label="FFT geglättet", color="blue")

# Harmonische der rohen FFT
for k in range(1, 12):
    hk = k * f0
    if hk > 1000:
        break
    plt.axvline(hk, color="red", linestyle="--", alpha=0.4)
    plt.text(hk, np.max(fft_mag)*0.9, f"{k}f0 roh", rotation=90, color="red")

# Harmonische der geglätteten FFT
for k in range(1, 12):
    hk = k * f0
    if hk > 1000:
        break
    plt.axvline(hk, color="green", linestyle="--", alpha=0.4)
    plt.text(hk, np.max(fft_smooth)*0.7, f"{k}f0 glatt", rotation=90, color="green")

plt.title(f"FFT roh + geglättet + Harmonische (f0 ≈ {f0:.1f} Hz)")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
