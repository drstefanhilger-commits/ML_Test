import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

def bandpass_sos(audio, sr, low=80, high=1500, order=4):
    nyq = 0.5 * sr
    low_n = low / nyq
    high_n = high / nyq
    print("low_n =", low_n, "high_n =", high_n)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfiltfilt(sos, audio)

folder = "./data/train_48k/drone_real_train"
files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
path = os.path.join(folder, files[0])

audio, sr = sf.read(path)
if audio.ndim > 1:
    audio = audio[:,0]

print("sr =", sr)
print("RMS vorher:", np.sqrt(np.mean(audio**2)))

audio_bp = bandpass_sos(audio, sr)
print("RMS nachher:", np.sqrt(np.mean(audio_bp**2)))

# FFT vorher/nachher
n = len(audio)
window = np.hanning(n)
freqs = np.fft.rfftfreq(n, 1/sr)

fft_before = np.fft.rfft(audio * window)
fft_after  = np.fft.rfft(audio_bp * window)

mag_before = np.abs(fft_before)
mag_after  = np.abs(fft_after)

plt.figure(figsize=(12,6))
plt.plot(freqs, mag_before, label="Vorher", color="gray", alpha=0.7)
plt.plot(freqs, mag_after,  label="Nachher (80–1500 Hz)", color="blue")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show()
