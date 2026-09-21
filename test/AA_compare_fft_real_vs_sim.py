import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def load_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:,0]
    return audio, sr

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
sim_folder  = "./data/train_48k/sim_drone_test"

# ---------------------------------------------------------
# Liste aller Dateien
# ---------------------------------------------------------
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])
sim_files  = sorted([f for f in os.listdir(sim_folder) if f.endswith(".wav")])

print(f"[INFO] Anzahl realer Dateien: {len(real_files)}")
print(f"[INFO] Anzahl simulierter Dateien: {len(sim_files)}")

# ---------------------------------------------------------
# Wähle eine Datei aus (Index anpassen!)
# ---------------------------------------------------------
REAL_INDEX = 0
SIM_INDEX  = 0

real_path = os.path.join(real_folder, real_files[REAL_INDEX])
sim_path  = os.path.join(sim_folder,  sim_files[SIM_INDEX])

print("[INFO] Real:", real_path)
print("[INFO] Sim :", sim_path)

# ---------------------------------------------------------
# Laden
# ---------------------------------------------------------
real_audio, sr_real = load_wav(real_path)
sim_audio,  sr_sim  = load_wav(sim_path)

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
# Plot 2: Simulated FFT
# ---------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(freqs_sim, mag_sim, label=f"Simulated: {sim_files[SIM_INDEX]}", color="orange")
plt.title("FFT – Simulated Drone")
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
plt.plot(freqs_sim,  mag_sim,  label=f"Simulated: {sim_files[SIM_INDEX]}", color="orange", alpha=0.7)
plt.title("FFT Comparison – Real vs Simulated Drone")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show(block=True)
