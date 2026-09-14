import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import random

INPUT_DIR = Path("data/train/no_drone")
OUTPUT_DIR = Path("data/train/no_drone_aug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

def load_wav(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    return y

def save_wav(path, y):
    sf.write(path, y, SAMPLE_RATE)

# -----------------------------
# Augmentations
# -----------------------------

def aug_gain(y):
    g = random.uniform(0.5, 1.5)
    return y * g

def aug_pitch(y):
    steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, SAMPLE_RATE, steps)

def aug_speed(y):
    rate = random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate)

def aug_noise(y):
    noise = np.random.randn(len(y)) * random.uniform(0.005, 0.05)
    return y + noise

def aug_bandpass(y):
    # simple bandpass via STFT mask
    S = librosa.stft(y)
    mag, phase = librosa.magphase(S)
    f = np.linspace(0, 1, mag.shape[0])
    mask = (f > random.uniform(0.05, 0.15)) & (f < random.uniform(0.6, 0.9))
    mag_filtered = mag * mask[:, None]
    return librosa.istft(mag_filtered * phase)

def aug_clip(y):
    c = random.uniform(0.7, 1.0)
    return np.clip(y, -c, c)

AUGS = [
    aug_gain,
    aug_pitch,
    aug_speed,
    aug_noise,
    aug_bandpass,
    aug_clip
]

# -----------------------------
# Main
# -----------------------------

def augment_file(path, count=50):
    y = load_wav(path)
    base = path.stem

    for i in range(count):
        y_aug = y.copy()
        # apply 2–4 random augmentations
        for _ in range(random.randint(2, 4)):
            aug = random.choice(AUGS)
            try:
                y_aug = aug(y_aug)
            except:
                continue

        out_path = OUTPUT_DIR / f"{base}_aug_{i}.wav"
        save_wav(out_path, y_aug)

    print(f"Augmented {path.name} -> {count} files")

def main():
    files = list(INPUT_DIR.glob("*.wav"))
    if not files:
        print("Keine NO-DRONE WAVs gefunden.")
        return

    print(f"Gefundene NO-DRONE Dateien: {len(files)}")
    for f in files:
        augment_file(f, count=50)

    print("\nAugmentierung abgeschlossen.")
    print("Neue Dateien liegen in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
