import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as sig

# ============================================================
# Magnitude bei f0 bestimmen
# ============================================================
def synth_freq_domain_harmonics_and_triangle(f0, freqs, K=6, tri_mag=1.0):
    """
    Erzeugt ein Frequenzspektrum mit:
    - reinen Harmonischen (Linien)
    - Dreieck von f0 bis 1000 Hz mit variabler Magnitude tri_mag
    """

    S = np.zeros_like(freqs)

    # Harmonische
    for k in range(1, K+1):
        fk = k * f0
        idx = np.argmin(np.abs(freqs - fk))
        S[idx] += 1.0 / k   # Amplitude

    # Dreieck
    T = np.zeros_like(freqs)
    f1 = f0
    f2 = 400.0

    for i, f in enumerate(freqs):
        if f1 <= f <= f2:
            T[i] = tri_mag * (1 - (f - f1) / (f2 - f1))

    return S + T, S, T

# ============================================================
# Magnitude bei f0 bestimmen
# ============================================================
def get_mag_at_f0(freqs, mag, f0, bw=2.0):
    mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    return np.max(mag[mask])


# ============================================================
# Reine Harmonische Simulation
# ============================================================
def simulate_harmonic_drone(f0, sr=48000, duration=2.0):
    N = int(sr * duration)
    t = np.linspace(0, duration, N, endpoint=False)

    K = 6
    s = np.zeros_like(t)

    for k in range(1, K+1):
        Ak = 1.0 / k
        s += Ak * np.sin(2 * np.pi * k * f0 * t)

    # keine Normalisierung – wir wollen echte Magnitude
    return s, sr


# ============================================================
# f0 der Simulation bestimmen (Peak-Interpolation)
# ============================================================
def detect_f0_sim(freqs, mag, f0_real):
    fmin = 0.5 * f0_real
    fmax = 1.5 * f0_real
    mask = (freqs >= fmin) & (freqs <= fmax)

    mag_win = mag[mask]
    freqs_win = freqs[mask]

    i = np.argmax(mag_win)

    if 1 <= i < len(mag_win)-1:
        alpha = mag_win[i-1]
        beta  = mag_win[i]
        gamma = mag_win[i+1]
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        f0_est = freqs_win[i] + p * (freqs_win[1] - freqs_win[0])
    else:
        f0_est = freqs_win[i]

    return f0_est


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


# ============================================================
# f0 aus realer Drohne bestimmen
# ============================================================
def detect_f0(freqs, mag, fmin=40, fmax=200):
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.argmax(mag[mask])
    return freqs[mask][idx]


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
# FFT real
# ---------------------------------------------------------
freqs_real, mag_real = compute_fft(real_audio, sr_real)


# ---------------------------------------------------------
# f0 bestimmen (real)
# ---------------------------------------------------------
f0_real = detect_f0(freqs_real, mag_real)
print(f"[INFO] Detected f0_real = {f0_real:.2f} Hz")


# ---------------------------------------------------------
# Simulation erzeugen
# ---------------------------------------------------------
sim_audio, sr_sim = simulate_harmonic_drone(f0_real, sr=sr_real, duration=len(real_audio)/sr_real)


# ---------------------------------------------------------
# FFT sim
# ---------------------------------------------------------
freqs_sim, mag_sim = compute_fft(sim_audio, sr_sim)


# ---------------------------------------------------------
# f0 bestimmen (sim)
# ---------------------------------------------------------
f0_sim_detected = detect_f0_sim(freqs_sim, mag_sim, f0_real)
print(f"[INFO] Detected f0_sim = {f0_sim_detected:.2f} Hz")


# ---------------------------------------------------------
# Magnitude Matching
# ---------------------------------------------------------
M_real = get_mag_at_f0(freqs_real, mag_real, f0_real)
M_sim  = get_mag_at_f0(freqs_sim,  mag_sim,  f0_real)

print(f"[INFO] M_real(f0) = {M_real:.3e}")
print(f"[INFO] M_sim(f0)  = {M_sim:.3e}")

scale = M_real / M_sim
print(f"[INFO] Scale factor = {scale:.3f}")

# Simulation skalieren
sim_audio *= scale
freqs_sim, mag_sim = compute_fft(sim_audio, sr_sim)

# Frequenzbereich-Synthese
S_total, S_harm, S_tri = synth_freq_domain_harmonics_and_triangle(
    f0_real,
    freqs_sim,
    K=6,
    tri_mag=0.5   # variable Magnitude
)



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
# Plot 2: Simulated FFT
# ---------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(freqs_sim, mag_sim, label=f"Simulated (f0={f0_real:.1f} Hz)", color="orange")
plt.title("FFT – Simulated Drone (Harmonic Model)")
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
plt.plot(freqs_real, mag_real, label="Real", color="blue")
plt.plot(freqs_sim,  mag_sim,  label="Simulated (scaled)", color="orange", alpha=0.7)
plt.title("FFT Comparison – Real vs Simulated (Magnitude matched)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=False)


# ---------------------------------------------------------
# Plot 4: Frequenzbereich-Synthese (Harmonische + Dreieck)
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(freqs_sim, S_harm, label="Simulated Harmonics", color="green")
plt.plot(freqs_sim, S_tri,  label="Triangle Noise", color="red", alpha=0.7)
plt.plot(freqs_sim, S_total, label="Combined Spectrum", color="black")
plt.title("Frequency-Domain Synthetic Spectrum (Harmonics + Triangle)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=True)
