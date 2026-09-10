import numpy as np
import tensorflow as tf
import librosa
import math
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
MODE = "A"   # A = reiner False-Test, B = Stress-Test
SDS_GAIN = 0.3   # Stärke der SDS-Drohnenüberlagerung (0.0–1.0)
SDS_NUM_MICS = 8
SDS_SAMPLE_RATE = 16000
SDS_FRAME_LEN = 256
c = 343.0

SDS_MIC_POSITIONS = np.array([
    [ 0.03,  0.03],
    [ 0.03, -0.03],
    [-0.03,  0.03],
    [-0.03, -0.03],
    [ 0.05,  0.00],
    [-0.05,  0.00],
    [ 0.00,  0.05],
    [ 0.00, -0.05]
])

mel_filterbank = np.load("mel_filterbank_40x129.npy")
TFLITE_PATH = "models/binary/model_binary_int8.tflite"

FALSE_WAV_DIR = Path("data/false_sounds")  # hier liegen Wind/Donner/Autos/Stimmen/etc.


# -----------------------------
# Fractional delay helper
# -----------------------------
def sample_with_delay(src, delay_samples, n):
    i0 = int(n - delay_samples)
    frac = (n - delay_samples) - i0

    if i0 < 0 or i0 + 1 >= len(src):
        return 0.0

    return (1 - frac) * src[i0] + frac * src[i0 + 1]


# -----------------------------
# Coherent noise source
# -----------------------------
def generate_source_noise():
    return np.random.randn(SDS_FRAME_LEN).astype(np.float32)


# -----------------------------
# Far-field SDS simulation
# -----------------------------
def make_sds_frame(angleDeg, distance_m, noise):
    theta = math.radians(angleDeg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    A = 1.0 / distance_m

    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)
    src = generate_source_noise()

    for ch in range(SDS_NUM_MICS):
        x, y = SDS_MIC_POSITIONS[ch]
        proj = x * dx + y * dy
        tau = proj / c
        delaySamples = tau * SDS_SAMPLE_RATE

        for n in range(SDS_FRAME_LEN):
            s = sample_with_delay(src, delaySamples, n)
            v = A * s

            if noise > 0.0:
                v += (np.random.rand() - 0.5) * noise * A

            mic[ch][n] = v

    return mic


# -----------------------------
# Mel extraction (STM32-compatible)
# -----------------------------
def stm32_mel_features(frame):
    window = np.hanning(SDS_FRAME_LEN)
    S = librosa.stft(
        frame,
        n_fft=SDS_FRAME_LEN,
        hop_length=SDS_FRAME_LEN // 2,
        window=window,
        center=False
    )

    mag = np.abs(S)
    mag = mag[:129, :]

    mel = mel_filterbank @ mag
    mel = np.log10(mel + 1e-6)

    feat = mel.mean(axis=1)
    return feat.astype(np.float32)


# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def infer_binary(feat):
    interpreter.set_tensor(input_details[0]['index'], feat.reshape(1, -1))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return float(out[0])

# -----------------------------
# Zero-Input Test (Modell-Diagnose)
# -----------------------------
zero = np.zeros(40, dtype=np.float32)
print("Zero-Input:", infer_binary(zero))


# -----------------------------
# WAV loader
# -----------------------------
def load_wav_mono(path):
    y, sr = librosa.load(path, sr=SDS_SAMPLE_RATE)
    if len(y) < SDS_FRAME_LEN:
        y = np.pad(y, (0, SDS_FRAME_LEN - len(y)))
    else:
        y = y[:SDS_FRAME_LEN]
    return y.astype(np.float32)


# -----------------------------
# Mix WAV + SDS
# -----------------------------
def mix_wav_and_sds(wav_frame, sds_frame, sds_gain=1.0):
    sds_mono = sds_frame.mean(axis=0)
    return wav_frame + sds_gain * sds_mono


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    wav_files = sorted(FALSE_WAV_DIR.glob("*.wav"))

    if not wav_files:
        print("Keine False-Test WAV-Dateien gefunden in:", FALSE_WAV_DIR)
        exit(1)

    print(f"Gefundene False-Test WAV-Dateien: {len(wav_files)}")
    print("Starte False-Test Analyse...\n")

    for wav_path in wav_files:
        print(f"\n=== FALSE-SOUND: {wav_path.name} ===")
        wav_frame = load_wav_mono(wav_path)

        if MODE == "A":
            # Reiner False-Test: keine SDS-Simulation
            mix = wav_frame
            feat = stm32_mel_features(mix)
            pred = infer_binary(feat)
            label = "DRONE" if pred > 0.5 else "NO DRONE"
            print(f"False-Test -> {pred:.3f}  ({label})")
            continue

        elif MODE == "B":
            # Stress-Test: SDS-Drohnenanteil überlagern
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
            distances = [30, 50, 70]
            noises = [0.0, 0.01, 0.05]

            for angle in angles:
                for dist in distances:
                    for noise in noises:
                        sds_frame = make_sds_frame(angle, dist, noise)
                        sds_mono = sds_frame.mean(axis=0)
                        mix = wav_frame + SDS_GAIN * sds_mono

                        feat = stm32_mel_features(mix)
                        pred = infer_binary(feat)
                        label = "DRONE" if pred > 0.5 else "NO DRONE"
                        print(f"Az={angle:3d}°  D={dist:3.0f}m  N={noise:.3f}  ->  {pred:.3f}  ({label})")
