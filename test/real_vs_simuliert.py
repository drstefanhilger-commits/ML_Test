import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, savgol_filter

# -----------------------------
# Bandpass
# -----------------------------
def bandpass_sos(audio, sr, low=80, high=600, order=4):
    nyq = 0.5 * sr
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, audio)

# -----------------------------
# f0-Erkennung
# -----------------------------
def detect_f0(freqs, mag, fmin=40, fmax=120):
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.argmax(mag[mask])
    return freqs[mask][idx]

# -----------------------------
# Nahfeld-Simulation
# -----------------------------
def simulate_nearfield_drone(sr=48000, duration=1.0, f0=65):
    t = np.linspace(0, duration, int(sr*duration))

    jitter = 0.02 * np.random.randn(len(t))
    f0_inst = f0 * (1 + jitter)

    sig = np.sin(2*np.pi * np.cumsum(f0_inst)/sr)
    for k in range(2, 8):
        sig += (1.0/k) * np.sin(2*np.pi * np.cumsum(k*f0_inst)/sr)

    am = 1 + 0.3*np.sin(2*np.pi*3*f0*t) + 0.1*np.random.randn(len(t))
    sig *= am

    noise = 0.05 * np.random.randn(len(t))
    sig += noise

    sig /= np.max(np.abs(sig)) + 1e-6
    return sig

# -----------------------------
# reale Drohne laden
# -----------------------------
real_folder = "./data/train_48k/drone_real_train"
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])
real_path = os.path.join(real_folder, real_files[0])

real_audio, sr_real = sf.read(real_path)
if real_audio.ndim > 1:
    real_audio = real_audio[:,0]

sr = sr_real

# -----------------------------
# Simulation erzeugen
# -----------------------------
sim_audio = simulate_nearfield_drone(sr=sr, duration=len(real_audio)/sr, f0=65)

# -----------------------------
# Bandpass
# -----------------------------
real_bp = bandpass_sos(real_audio, sr)
sim_bp  = bandpass_sos(sim_audio,  sr)

# -----------------------------
# FFT
# -----------------------------
n = min(len(real_bp), len(sim_bp))
window = np.hanning(n)
freqs = np.fft.rfftfreq(n, 1/sr)

fft_real = np.abs(np.fft.rfft(real_bp[:n] * window))
fft_sim  = np.abs(np.fft.rfft(sim_bp[:n]  * window))

# Glättung
fft_real_smooth = savgol_filter(fft_real, 101, 3)
fft_sim_smooth  = savgol_filter(fft_sim,  101, 3)

# f0
f0_real = detect_f0(freqs, fft_real)
f0_sim  = detect_f0(freqs, fft_sim)

# -----------------------------
# Plot 1: FFT roh + Harmonische
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(freqs, fft_real, label="Real – roh", color="blue", alpha=0.6)
plt.plot(freqs, fft_sim,  label="Sim Nahfeld – roh",  color="orange", alpha=0.6)

for k in range(1, 12):
    hr = k * f0_real
    hs = k * f0_sim
    if hr > 1000 and hs > 1000:
        break
    if hr <= 1000:
        plt.axvline(hr, color="blue", linestyle="--", alpha=0.4)
    if hs <= 1000:
        plt.axvline(hs, color="orange", linestyle="--", alpha=0.4)

plt.title("FFT – reale Drohne vs. Nahfeld-Simulation (roh, mit Harmonischen)")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: FFT geglättet + Harmonische
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(freqs, fft_real_smooth, label="Real – glatt", color="blue")
plt.plot(freqs, fft_sim_smooth,  label="Sim Nahfeld – glatt",  color="green")

for k in range(1, 12):
    hr = k * f0_real
    hs = k * f0_sim
    if hr > 1000 and hs > 1000:
        break
    if hr <= 1000:
        plt.axvline(hr, color="blue", linestyle="--", alpha=0.4)
    if hs <= 1000:
        plt.axvline(hs, color="green", linestyle="--", alpha=0.4)

plt.title("FFT geglättet – reale Drohne vs. Nahfeld-Simulation (mit Harmonischen)")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
