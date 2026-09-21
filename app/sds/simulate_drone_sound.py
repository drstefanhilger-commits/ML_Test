import numpy as np
import scipy.signal as sig
import soundfile as sf

# ============================================================
# Parameter
# ============================================================
sr = 48000               # Sample Rate
duration = 2.0           # Sekunden
N = int(sr * duration)   # Samples
f0 = 120                 # Grundfrequenz des Rotors
K = 8                    # Anzahl Harmonischer
sigma = 15               # Verbreiterung der Harmonischen (Hz)

# Bandpass-Filter (Sound wird danach sowieso begrenzt)
bp_low = 80
bp_high = 1500

# ============================================================
# Frequenzachse
# ============================================================
freqs = np.fft.rfftfreq(N, 1/sr)
S = np.zeros_like(freqs, dtype=np.complex128)

# ============================================================
# Harmonische erzeugen
# ============================================================
for k in range(1, K+1):
    fk = k * f0
    Ak = 1.0 / k               # Amplitude abnehmend
    Hk = Ak * np.exp(-(freqs - fk)**2 / (2 * sigma**2))
    S += Hk

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
# IFFT → Zeitbereich
# ============================================================
s = np.fft.irfft(S)
s /= np.max(np.abs(s))   # Normalisieren

# ============================================================
# Bandpass-Filter anwenden
# ============================================================
sos = sig.butter(4, [bp_low/(sr/2), bp_high/(sr/2)], btype='band', output='sos')
s_bp = sig.sosfiltfilt(sos, s)

# ============================================================
# Speichern
# ============================================================
sf.write("drone_sim.wav", s_bp, sr)
print("Fertig: drone_sim.wav erzeugt.")
