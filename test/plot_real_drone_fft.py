import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Neuer f0-Suchbereich
F0_MIN = 40
F0_MAX = 120
N_HARM = 12

def detect_f0(freqs, mag):
    mask = (freqs >= F0_MIN) & (freqs <= F0_MAX)
    idx = np.argmax(mag[mask])
    f0 = freqs[mask][idx]
    return f0

def main():
    # folder = "./data/train_48k/drone_real_test"
    folder = "./data/train_48k/drone_sim_test"
    files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]

    if len(files) == 0:
        raise Exception("Keine WAV-Dateien im Ordner gefunden.")

    path = os.path.join(folder, files[12])
    print(f"[INFO] Lade Datei: {path}")

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:,0]

    n = len(audio)
    window = np.hanning(n)
    fft = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(n, 1/sr)
    mag = np.abs(fft)

    # f0 erkennen
    f0 = detect_f0(freqs, mag)
    print(f"[INFO] Erkanntes f0: {f0:.2f} Hz")

    # Harmoniken berechnen
    harmonics = [k * f0 for k in range(1, N_HARM + 1)]
    print("[INFO] Harmoniken:")
    for k, hk in enumerate(harmonics, start=1):
        print(f"  {k} * f0 = {hk:.2f} Hz")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(freqs, mag, label="Magnitude Spectrum")

    # f0 markieren
    plt.axvline(f0, color="green", linestyle="--", alpha=0.8)
    plt.text(f0, np.max(mag)*0.9, f"f0={f0:.1f} Hz", rotation=90, color="green")

    # Harmoniken markieren
    for hk in harmonics:
        if hk <= 1000:
            plt.axvline(hk, color="red", linestyle="--", alpha=0.6)
            plt.text(hk, np.max(mag)*0.8, f"{hk:.0f} Hz", rotation=90, color="red")

    plt.title(f"FFT – Real Drone Sample: {files[0]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()

    # Nur 0–1000 Hz anzeigen
    plt.xlim(0, 1000)

    plt.show()

if __name__ == "__main__":
    main()
