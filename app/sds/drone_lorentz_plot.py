import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import soundfile as sf

# ============================================================
# Parameter
# ============================================================
sr = 48000               # Sample Rate
duration = 2.0           # Sekunden
N = int(sr * duration)   # Samples
f0 = 120                 # Grundfrequenz des Rotors
K = 8                    # Anzahl Harmonischer
gamma = 20               # Lorentz-Breite

# Bandpass-Filter
bp_low = 80
bp_high = 1500

# ============================================================
# Frequenzachse
# ============================================================
freqs = np.fft.rfftfreq(N, 1/sr)
S = np.zeros_like(freqs, dtype=np.complex128)

# ============================================================
# Lorentz-Harmonische erzeugen
# ============================================================
for k in range(1, K+1):
    fk = k * f0
    Ak = 1.0 / k
    Lk = Ak / (1 + ((freqs - fk) / gamma)**2)
    S += Lk

# ============================================================
# Dreiecksrauschen (100 Hz → 80, 1000 Hz → 0)
# ============================================================
T = np.zeros_like(freqs)

f1 = 100
f2 = 1000
for i, f in enumerate(freqs):
    if f1 <= f <= f2:
        T[i] = 80 * (1 - (f - f1) / (f2 - f1))

# Weißes Rauschen im Frequenzbereich
Nf = (np.random.randn(len(freqs)) + 1j*np.random.randn(len(freqs)))
Nf *= T

S += Nf

# ============================================================
# Plot des Spektrums
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(freqs, np.abs(S), label="Spektrum (Lorentz + Rauschen)")
plt.title("Simuliertes Drohnenspektrum mit Lorentz-Harmonischen")
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# IFFT → Zeitbereich
# ============================================================
s = np.fft.irfft(S)
s /= np.max(np.abs(s))

# ============================================================
# Bandpass-Filter anwenden
# ============================================================
sos = sig.butter(4, [bp_low/(sr/2), bp_high/(sr/2)], btype='band', output='sos')
s_bp = sig.sosfiltfilt(sos, s)

# ============================================================
# Speichern
# ============================================================
sf.write("drone_lorentz.wav", s_bp, sr)
print("Fertig: drone_lorentz.wav erzeugt.")
