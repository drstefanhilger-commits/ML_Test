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

# ----------------------------------------
# Import synthetic SDS drone source
# ----------------------------------------
from sds_drone_source import generate_sds_drone_source

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
# SDS far-field simulation (8 channels)
# ----------------------------------------
def make_sds_frame(src_frame, angleDeg, distance_m, noise):
    theta = math.radians(angleDeg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    A = 1.0 / distance_m

    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)

    for ch in range(SDS_NUM_MICS):
        x, y = SDS_MIC_POSITIONS[ch]
        proj = x * dx + y * dy
        tau = proj / c
        delaySamples = tau * SDS_SAMPLE_RATE

        for n in range(SDS_FRAME_LEN):
            s = sample_with_delay(src_frame, delaySamples, n)
            v = A * s

            if noise > 0.0:
                v += (np.random.rand() - 0.5) * noise * A

            mic[ch][n] = v

    return mic

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
# Load ML model
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
# MAIN TEST PROGRAM
# ----------------------------------------
if __name__ == "__main__":
    print("\n=== SDS Synthetic Drone Test ===\n")

    # 1) Synthetic Drone Source
    drone_src = generate_sds_drone_source()

    # 2) Synthetic Non-Drone Source (white noise)
    no_drone_src = np.random.randn(SDS_FRAME_LEN).astype(np.float32)
    no_drone_src /= np.max(np.abs(no_drone_src) + 1e-6)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [10, 20, 30]
    noises = [0.00, 0.01, 0.05]

    for angle in angles:
        for dist in distances:
            for noise in noises:

                # --- Drone ---
                sds_drone = make_sds_frame(drone_src, angle, dist, noise)
                mono_drone = sds_drone.mean(axis=0)
                feat_drone = stm32_mel_features(mono_drone)
                pred_drone = infer_binary(feat_drone)

                # --- No Drone ---
                sds_no = make_sds_frame(no_drone_src, angle, dist, noise)
                mono_no = sds_no.mean(axis=0)
                feat_no = stm32_mel_features(mono_no)
                pred_no = infer_binary(feat_no)

                print(f"Az={angle:3d}° D={dist:3d}m N={noise:.3f} | "
                      f"Drone={pred_drone:.3f}  NoDrone={pred_no:.3f}")
