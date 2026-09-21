import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, savgol_filter

def match_sim_to_real(freqs, real_fft, sim_fft, smooth_win=51):
    """
    Passt die simulierte Drohne automatisch an die reale Drohne an.
    - Harmonische Energie angleichen
    - Peakbreite angleichen
    - Noise-Floor angleichen
    - globale Skalierung
    """

    sim_adj = sim_fft.copy()

    # --- f0 bestimmen ---
    def detect_f0(freqs, mag, fmin=40, fmax=120):
        mask = (freqs >= fmin) & (freqs <= fmax)
        idx = np.argmax(mag[mask])
        return freqs[mask][idx]

    f0_real = detect_f0(freqs, real_fft)
    f0_sim  = detect_f0(freqs, sim_fft)

    # --- Harmonische bestimmen ---
    harmonics_real = [k * f0_real for k in range(1, 12)]
    harmonics_sim  = [k * f0_sim  for k in range(1, 12)]

    # --- Harmonische Energie angleichen ---
    for hr, hs in zip(harmonics_real, harmonics_sim):

        idx_real = np.argmin(np.abs(freqs - hr))
        idx_sim  = np.argmin(np.abs(freqs - hs))

        ratio = (real_fft[idx_real] + 1e-6) / (sim_fft[idx_sim] + 1e-6)

        # lokale Peakregion verstärken
        sim_adj[idx_sim-3:idx_sim+4] *= ratio

    # --- Peakbreite angleichen ---
    sim_adj = savgol_filter(sim_adj, smooth_win, 3)

    # --- Noise-Floor angleichen ---
    noise_real = np.median(real_fft[200:400])
    noise_sim  = np.median(sim_adj[200:400])
    sim_adj += (noise_real - noise_sim) * 0.5

    # --- globale Skalierung ---
    idx_real_f0 = np.argmin(np.abs(freqs - f0_real))
    idx_sim_f0  = np.argmin(np.abs(freqs - f0_real))

    scale = real_fft[idx_real_f0] / (sim_adj[idx_sim_f0] + 1e-6)
    sim_adj *= scale

    return sim_adj


def triangular_filter_inv(freqs, f_low=100, f_high=1000, A=1.0):
    H = np.zeros_like(freqs)

    # Bereich 1: unter f_low -> 0
    H[freqs <= f_low] = 0

    # Bereich 2: linearer ANSTIEG
    mask = (freqs > f_low) & (freqs < f_high)
    H[mask] = A * ((freqs[mask] - f_low) / (f_high - f_low))

    # Bereich 3: über f_high -> A
    H[freqs >= f_high] = A

    return H

def triangular_filter(freqs, f_low=100, f_high=1000, A=1.0):
    H = np.zeros_like(freqs)

    # Bereich 1: unter f_low -> volle Intensität A
    H[freqs <= f_low] = A

    # Bereich 2: linearer Abfall
    mask = (freqs > f_low) & (freqs < f_high)
    H[mask] = A * (1 - (freqs[mask] - f_low) / (f_high - f_low))

    # Bereich 3: über f_high -> 0
    H[freqs >= f_high] = 0

    return H

# ---------------------------------------------------------
# Bandpass
# ---------------------------------------------------------
def bandpass_sos(audio, sr, low=80, high=600, order=4):
    nyq = 0.5 * sr
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, audio)

# ---------------------------------------------------------
# f0-Erkennung
# ---------------------------------------------------------
def detect_f0(freqs, mag, fmin=40, fmax=120):
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.argmax(mag[mask])
    return freqs[mask][idx]

# ---------------------------------------------------------
# Nahfeld-Simulation mit logarithmischer Gewichtung
# ---------------------------------------------------------
def simulate_nearfield_drone(sr, duration, f0, alpha=1.4):
    t = np.linspace(0, duration, int(sr * duration))

    jitter = 0.01 * np.random.randn(len(t))
    f0_inst = f0 * (1 + jitter)

    sig = 1.0 * np.sin(2 * np.pi * np.cumsum(f0_inst) / sr)

    max_harm = 6
    for k in range(2, max_harm + 1):
        w = 1.0 / (k ** alpha)
        sig += w * np.sin(2 * np.pi * np.cumsum(k * f0_inst) / sr)

    am = 1 + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    sig *= am

    noise = 0.02 * np.random.randn(len(t))
    sig += noise

    sig /= np.max(np.abs(sig)) + 1e-6
    return sig

# ---------------------------------------------------------
# Reale Drohne laden
# ---------------------------------------------------------
real_folder = "./data/train_48k/drone_real_train"
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])
real_path = os.path.join(real_folder, real_files[0])

real_audio, sr = sf.read(real_path)
if real_audio.ndim > 1:
    real_audio = real_audio[:,0]

# ---------------------------------------------------------
# Bandpass real
# ---------------------------------------------------------
real_bp = bandpass_sos(real_audio, sr)

# ---------------------------------------------------------
# FFT real
# ---------------------------------------------------------
n = len(real_bp)
window = np.hanning(n)
freqs = np.fft.rfftfreq(n, 1/sr)

fft_real = np.abs(np.fft.rfft(real_bp * window))
fft_real_smooth = savgol_filter(fft_real, 101, 3)

# ---------------------------------------------------------
# f0 real bestimmen
# ---------------------------------------------------------
f0_real = detect_f0(freqs, fft_real)
print("f0 real:", f0_real)

# ---------------------------------------------------------
# Simulation erzeugen
# ---------------------------------------------------------
sim_audio = simulate_nearfield_drone(sr, duration=n/sr, f0=f0_real)

# ---------------------------------------------------------
# Bandpass sim
# ---------------------------------------------------------
sim_bp = bandpass_sos(sim_audio, sr)

# ---------------------------------------------------------
# FFT sim
# ---------------------------------------------------------
fft_sim = np.abs(np.fft.rfft(sim_bp * window))
fft_sim_smooth = savgol_filter(fft_sim, 101, 3)

fft_sim_matched = match_sim_to_real(freqs, fft_real, fft_sim)
fft_sim_matched_smooth = savgol_filter(fft_sim_matched, 101, 3)


# ---------------------------------------------------------
# f0-Peaks vergleichen und Simulation skalieren
# ---------------------------------------------------------
idx_real_f0 = np.argmin(np.abs(freqs - f0_real))
idx_sim_f0  = np.argmin(np.abs(freqs - f0_real))

I_real_f0 = fft_real[idx_real_f0]
I_sim_f0  = fft_sim[idx_sim_f0]

scale = I_real_f0 / I_sim_f0
print("Skalierungsfaktor:", scale)

# Simulation skalieren
sim_audio *= scale

# ---------------------------------------------------------
# Bandpass + FFT nach Skalierung
# ---------------------------------------------------------
sim_bp = bandpass_sos(sim_audio, sr)
fft_sim = np.abs(np.fft.rfft(sim_bp * window))
fft_sim_smooth = savgol_filter(fft_sim, 101, 3)

# ---------------------------------------------------------
# Dreieckfilter erzeugen (nur Simulation!)
# ---------------------------------------------------------
H_sim = triangular_filter(freqs, f_low=100, f_high=1000, A=1.0)

# Filter auf Simulation anwenden (korrekt NACH Skalierung!)
fft_sim_filtered = fft_sim * H_sim
fft_sim_filtered_smooth = savgol_filter(fft_sim_filtered, 101, 3)

# Zeitbereich zurückrechnen
sim_filtered = np.fft.irfft(fft_sim_filtered)


# Glättung
fft_sim_filtered_smooth = savgol_filter(fft_sim_filtered, 101, 3)

sim_filtered = np.fft.irfft(fft_sim_filtered)

# ---------------------------------------------------------
# Bandpass + FFT nach Skalierung
# ---------------------------------------------------------
sim_bp = bandpass_sos(sim_audio, sr)
fft_sim = np.abs(np.fft.rfft(sim_bp * window))
fft_sim_smooth = savgol_filter(fft_sim, 101, 3)


# Dreieckfilter erzeugen
H = triangular_filter(freqs, f_low=100, f_high=1000, A=1.0)

# Filter auf reales Spektrum anwenden
fft_real_filtered = fft_real * H

# Glättung
fft_real_filtered_smooth = savgol_filter(fft_real_filtered, 101, 3)




plt.figure(figsize=(10,4))
plt.plot(freqs, H, color="red")
plt.title("Dreieckiger Frequenzfilter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain")
plt.grid(True)
plt.tight_layout()
plt.show(block=False)



# ---------------------------------------------------------
# Plot 1: FFT roh
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(freqs, fft_real, label="Real – roh", color="blue", alpha=0.4)
plt.plot(freqs, fft_real_filtered, label="Real – Dreieck-gefiltert", color="red", alpha=0.8)
plt.plot(freqs, fft_sim,  label="Sim – roh (skaliert)", color="orange", alpha=0.6)

for k in range(1, 12):
    hr = k * f0_real
    if hr > 1000:
        break
    plt.axvline(hr, color="gray", linestyle="--", alpha=0.3)

plt.title("FFT roh – Real vs. Dreieck-gefiltert vs. Simulation")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ---------------------------------------------------------
# Plot 2: FFT geglättet
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(freqs, fft_real_smooth, label="Real – glatt", color="blue")
plt.plot(freqs, fft_real_filtered_smooth, label="Real – Dreieck-gefiltert – glatt", color="red")
plt.plot(freqs, fft_sim_smooth,  label="Sim – glatt (skaliert)", color="green")

for k in range(1, 12):
    hr = k * f0_real
    if hr > 1000:
        break
    plt.axvline(hr, color="gray", linestyle="--", alpha=0.3)

plt.title("FFT geglättet – Real vs. Dreieck-gefiltert vs. Simulation")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ---------------------------------------------------------
# Plot 3: Überlagerte Zeitsignale (Real vs Simulation)
# ---------------------------------------------------------
plt.figure(figsize=(12,6))

# Nur die ersten 0.1 Sekunden zeigen (besser sichtbar)
t = np.arange(n) / sr
mask = t < 0.1

plt.plot(t[mask], real_bp[mask], label="Real – Zeitbereich", color="blue", alpha=0.7)
plt.plot(t[mask], sim_bp[mask],  label="Sim – Zeitbereich (skaliert)", color="orange", alpha=0.7)

plt.title("Überlagerte Zeitsignale – Real vs. Simulation")
plt.xlabel("Zeit (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


plt.figure(figsize=(12,6))
plt.plot(freqs, fft_real, label="Real", color="blue")
plt.plot(freqs, fft_sim_matched_smooth, label="Sim (angepasst)", color="green")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=False)


plt.figure(figsize=(12,6))
plt.plot(freqs, fft_sim, label="Sim – roh", color="orange", alpha=0.5)
plt.plot(freqs, fft_sim_filtered, label="Sim – Dreieck-gefiltert", color="red", alpha=0.8)

plt.title("FFT – Simulation roh vs. Dreieck-gefiltert")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1500)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)


plt.figure(figsize=(12,6))
plt.plot(freqs, fft_sim_smooth, label="Sim – glatt", color="green")
plt.plot(freqs, fft_sim_filtered_smooth, label="Sim – Dreieck-gefiltert – glatt", color="red")

plt.title("FFT geglättet – Simulation vs. Dreieck-gefiltert")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 1500)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
