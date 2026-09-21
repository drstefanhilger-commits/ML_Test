import os
import numpy as np
import soundfile as sf

# ---------------------------------------------------------
# SDS PARAMETERS
# ---------------------------------------------------------
SR = 48000
N_FFT = 4096
HOP = 2048
BANDS = 64

DURATION = 2.0
N_SAMPLES = 160

F0_MIN = 80
F0_MAX = 120
N_HARM = 12

HARM_BW = 50.0
AM_DEPTH = 0.15
AM_RATE = 6.0
RPM_JITTER = 0.02
NOISE_LEVEL = 0.08


def generate_exact_spectral_drone():
    n_frames = int((SR * DURATION) / HOP)

    # KORREKTE AUDIO-LÄNGE
    audio_length = (n_frames - 1) * HOP + N_FFT
    audio = np.zeros(audio_length)

    spectrum = np.zeros((N_FFT//2 + 1, n_frames))

    f0 = np.random.uniform(F0_MIN, F0_MAX)

    for k in range(1, N_HARM + 1):
        fk = k * f0
        amp = 1.0 / (k ** 0.8)

        for frame in range(n_frames):
            am = 1.0 + AM_DEPTH * np.sin(2 * np.pi * AM_RATE * frame / n_frames)
            jitter_freq = fk * (1 + RPM_JITTER * np.random.randn())

            freqs = np.arange(N_FFT//2 + 1)
            spectrum[:, frame] += amp * am * np.exp(
                -0.5 * ((freqs - jitter_freq) / HARM_BW) ** 2
            )

    spectrum += NOISE_LEVEL * np.random.rand(*spectrum.shape)

    # Overlap-Add
    for frame in range(n_frames):
        mag = spectrum[:, frame]
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(mag)))
        stft_frame = mag * phase

        frame_td = np.fft.irfft(stft_frame, n=N_FFT)

        start = frame * HOP
        audio[start:start+N_FFT] += frame_td

    audio /= np.max(np.abs(audio)) + 1e-9
    return audio


def main():
    OUT_DIR = "./data/train_48k/sim_drone_test"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Lösche alte WAV-Dateien in {OUT_DIR}")
    for f in os.listdir(OUT_DIR):
        if f.lower().endswith(".wav"):
            os.remove(os.path.join(OUT_DIR, f))
    print("[INFO] Alte Dateien gelöscht.")

    print(f"[INFO] Erzeuge {N_SAMPLES} exakt SDS-kompatible Sim-Drohnen")

    for i in range(N_SAMPLES):
        audio = generate_exact_spectral_drone()
        sf.write(os.path.join(OUT_DIR, f"sim_drone_{i:03d}.wav"), audio, SR)

    print("[INFO] Fertig.")


if __name__ == "__main__":
    main()
