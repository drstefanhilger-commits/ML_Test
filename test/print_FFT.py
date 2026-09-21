import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

folder = "./data/train_48k/drone_real_train"
files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]

if len(files) == 0:
    raise Exception("Keine WAV-Dateien im Ordner gefunden.")

path = os.path.join(folder, files[0])

audio, sr = sf.read(path)
if audio.ndim > 1:
    audio = audio[:,0]

n = len(audio)
window = np.hanning(n)
fft = np.fft.rfft(audio * window)
freqs = np.fft.rfftfreq(n, 1/sr)

plt.figure(figsize=(12,6))
plt.plot(freqs, np.abs(fft))
plt.title("FFT – Real Drone Sample: " + files[0])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
