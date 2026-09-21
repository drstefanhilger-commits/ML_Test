import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as sig

# ============================================================
# Simulation: Lorentz-Harmonische + Dreiecksrauschen
# ============================================================
def simulate_lorentz_drone(sr=48000, duration=2.0):
    N = int(sr * duration)
    freqs = np.fft.rfftfreq(N, 1/sr)

    # Parameter
    f0 = 120        # Grundfrequenz
    K = 8           # Anzahl Harmonischer
    gamma = 20      # Lorentz-Breite

    # Spektrum
    S = np.zeros_like(freqs, dtype=np.complex128)

    # Lorentz-Harmonische
    for k in range(1, K+1):
        fk = k * f0
        Ak = 1.0 / k
        Lk = Ak / (1 + ((freqs - fk) / gamma)**2)
        S += Lk

    # Dreiecksrauschen
    T = np.zeros_like(freqs)
    f1, f2 = 100, 1000
    for i, f in enumerate(freqs):
        if f1 <= f <= f2:
            T[i] = 80 * (1 - (f - f1) / (f2 - f1))

    Nf = (np.random.randn(len(freqs)) + 1j*np.random.randn(len(freqs)))
    Nf *= T
    S += Nf

    # IFFT → Zeitbereich
    s = np.fft.irfft(S)
    s /= np.max(np.abs(s))

    # Bandpass
    bp_low = 80
    bp_high = 1500
    sos = sig.butter(4, [bp_low/(sr/2), bp_high/(sr/2)], btype='band', output='sos')
    s_bp = sig.sosfiltfilt(sos, s)

    return s_bp, sr


# ============================================================
# WAV laden
# ============================================================
def load_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:,0]
    return audio, sr


# ============================================================
# FFT
# ============================================================
def compute_fft(audio, sr):
    n = len(audio)
    window = np.hanning(n)
    fft = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(n, 1/sr)
    mag = np.abs(fft)
    return freqs, mag


# ---------------------------------------------------------
# Ordnerpfade
# ---------------------------------------------------------
real_folder = "./data/train_48k/drone_real_train"
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])

print(f"[INFO] Anzahl realer Dateien: {len(real_files)}")

REAL_INDEX = 0
real_path = os.path.join(real_folder, real_files[REAL_INDEX])
print("[INFO] Real:", real_path)

# ---------------------------------------------------------
# Real laden
# ---------------------------------------------------------
real_audio, sr_real = load_wav(real_path)

# ---------------------------------------------------------
# Simulation erzeugen (anstelle WAV)
# ---------------------------------------------------------
sim_audio, sr_sim = simulate_lorentz_drone(sr=sr_real, duration=len(real_audio)/sr_real)

# ---------------------------------------------------------
# FFT berechnen
# ---------------------------------------------------------
freqs_real, mag_real = compute_fft(real_audio, sr_real)
freqs_sim,  mag_sim  = compute_fft(sim_audio, sr_sim)

# ---------------------------------------------------------
# Plot 1: Real FFT
# ---------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(freqs_real, mag_real, label=f"Real: {real_files[REAL_INDEX]}", color="blue")
plt.title("FFT – Real Drone")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=False)

# ---------------------------------------------------------
# Plot 2: Simulated FFT (Lorentz)
# ---------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(freqs_sim, mag_sim, label="Simulated (Lorentz)", color="orange")
plt.title("FFT – Simulated Drone (Lorentz Model)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=False)

# ---------------------------------------------------------
# Plot 3: Beide übereinander
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(freqs_real, mag_real, label=f"Real: {real_files[REAL_INDEX]}", color="blue")
plt.plot(freqs_sim,  mag_sim,  label="Simulated (Lorentz)", color="orange", alpha=0.7)
plt.title("FFT Comparison – Real vs Lorentz Simulation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=True)
