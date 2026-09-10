import numpy as np
import tensorflow as tf
import math
import librosa
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
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

WAV_DIR = Path("data/selected")  # echte WAVs hier

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
# Mix real WAV + SDS simulation
# -----------------------------
def load_wav_mono(path, target_len=SDS_FRAME_LEN):
    y, sr = librosa.load(path, sr=SDS_SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)

def mix_wav_and_sds(wav_frame, sds_frame, sds_gain=1.0):
    # einfache Mischung: WAV + gemitteltes SDS-Signal
    sds_mono = sds_frame.mean(axis=0)
    mix = wav_frame + sds_gain * sds_mono
    return mix

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [30, 50, 70]
    noises = [0.0, 0.01, 0.05]

    wav_files = sorted(WAV_DIR.glob("*.wav"))

    if not wav_files:
        print("Keine WAV-Dateien in", WAV_DIR)
        exit(1)

    print(f"Gefundene WAV-Dateien: {len(wav_files)}")
    print("Untersuche Kombinationen aus WAV + SDS (Winkel, Entfernung, Rauschen):\n")

    for wav_path in wav_files:
        print(f"\n=== WAV: {wav_path.name} ===")
        wav_frame = load_wav_mono(wav_path)

        for angle in angles:
            for dist in distances:
                for noise in noises:
                    sds_frame = make_sds_frame(angle, dist, noise)
                    mix = mix_wav_and_sds(wav_frame, sds_frame, sds_gain=1.0)

                    feat = stm32_mel_features(mix)
                    pred = infer_binary(feat)

                    label = "DRONE" if pred > 0.5 else "NO DRONE"
                    print(f"Az={angle:3d}°  D={dist:3.0f}m  N={noise:.3f}  ->  {pred:.3f}  ({label})")
