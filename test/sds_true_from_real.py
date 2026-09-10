import numpy as np
import tensorflow as tf
import math
import librosa
from pathlib import Path

# ----------------------------------------
# CONFIG
# ----------------------------------------
SDS_NUM_MICS = 8
SDS_SAMPLE_RATE = 16000
SDS_FRAME_LEN = 4096
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

DRONE_DIR = Path("data/train/drone")   # echte Drone-WAVs

# ----------------------------------------
# Fractional delay helper
# ----------------------------------------
def sample_with_delay(src, delay_samples, n):
    i0 = int(n - delay_samples)
    frac = (n - delay_samples) - i0

    if i0 < 0 or i0 + 1 >= len(src):
        return 0.0

    return (1 - frac) * src[i0] + frac * src[i0 + 1]

# ----------------------------------------
# Real drone source (WAV)
# ----------------------------------------
def load_drone_source(path, target_len=SDS_FRAME_LEN):
    y, sr = librosa.load(path, sr=SDS_SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)

# ----------------------------------------
# SDS far-field simulation aus echter Drone-WAV
# ----------------------------------------
def make_sds_drone_from_real_wav(drone_frame, angleDeg, distance_m, noise):
    theta = math.radians(angleDeg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    A = 1.0 / distance_m

    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)
    src = drone_frame  # echte Drone als Quelle

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

    # Mono für dein Binary-Modell
    return mic.mean(axis=0)

# ----------------------------------------
# Mel extraction
# ----------------------------------------
def stm32_mel_features(frame):
    window = np.hanning(SDS_FRAME_LEN)
    S = librosa.stft(
        frame,
        n_fft=SDS_FRAME_LEN,
        hop_length=SDS_FRAME_LEN // 2,
        window=window,
        center=False
    )

    mag = np.abs(S)[:129, :]
    mel = mel_filterbank @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)

# ----------------------------------------
# Load TFLite model
# ----------------------------------------
interpreter = tf.lite.Interpreter(TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def infer_binary(feat):
    interpreter.set_tensor(input_details[0]['index'], feat.reshape(1, -1))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return float(out[0])

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    drone_files = sorted(DRONE_DIR.glob("*.wav"))
    if not drone_files:
        print("Keine Drone-WAVs in", DRONE_DIR)
        exit(1)

    # nimm eine repräsentative echte Drone
    drone_path = drone_files[0]
    print("Verwende Drone-WAV:", drone_path.name)
    drone_frame = load_drone_source(drone_path)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [10, 20, 30, 50]
    noises = [0.00, 0.01, 0.05]

    print("\nStarte SDS-True-Test mit echter Drone-WAV...\n")

    for angle in angles:
        for dist in distances:
            for noise in noises:
                sds = make_sds_drone_from_real_wav(drone_frame, angle, dist, noise)
                feat = stm32_mel_features(sds)
                pred = infer_binary(feat)
                label = "DRONE" if pred > 0.5 else "NO DRONE"

                print(f"Az={angle:3d}°  D={dist:3d}m  N={noise:.3f}  ->  {pred:.3f}  ({label})")
