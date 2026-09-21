import os
import numpy as np
import soundfile as sf

SR = 48000
DURATION = 2.0
N_SAMPLES = 160

N_FFT = 4096
HOP = 2048

F0_MIN = 80
F0_MAX = 120
N_HARM = 12

BANDWIDTH = 40      # realistische Bandbreite
AM_DEPTH = 0.15
RPM_JITTER = 0.02

NOISE_LEVEL = 0.1


def generate_spectral_drone():
    n_frames = int((SR * DURATION) / HOP)
    spectrum = np.zeros((N_FFT//2 + 1, n_frames))

    f0 = np.random.uniform(F0_MIN, F0_MAX)

    for k in range(1, N_HARM + 1):
        fk = k * f0

        # Bandbreite modellieren
        bw = BANDWIDTH * (1 + 0.3 * np.random.randn())

        # Amplitude
        amp = 1.0 / k

        for frame in range(n_frames):
            # AM-Modulation
            am = 1.0 + AM_DEPTH * np.sin(2 * np.pi * 4 * frame / n_frames)

            # RPM-Jitter
            jitter = fk * (1 + RPM_JITTER * np.random.randn())

            # Gaußsche Harmoniken
            freqs = np.arange(N_FFT//2 + 1)
            spectrum[:, frame] += amp * am * np.exp(-0.5 * ((freqs - jitter) / bw)**2)

    # Noise-Floor
    spectrum += NOISE_LEVEL * np.random.rand(*spectrum.shape)

    # Zeitbereich rekonstruieren
    audio = np.zeros(int(SR * DURATION))
    frame_idx = 0

    for frame in range(n_frames):
        mag = spectrum[:, frame]
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag)))
        stft_frame = mag * phase

        frame_td = np.fft.irfft(stft_frame, n=N_FFT)
        start = frame_idx * HOP
        audio[start:start+N_FFT] += frame_td
        frame_idx += 1

    audio /= np.max(np.abs(audio)) + 1e-9
    return audio


def main():
    OUT_DIR = "./data/train_48k/sim_drone_test"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Erzeuge {N_SAMPLES} spektrale Sim-Drohnen in {OUT_DIR}")

    for i in range(N_SAMPLES):
        audio = generate_spectral_drone()
        sf.write(os.path.join(OUT_DIR, f"sim_drone_{i:03d}.wav"), audio, SR)

    print("[INFO] Fertig.")


if __name__ == "__main__":
    main()
