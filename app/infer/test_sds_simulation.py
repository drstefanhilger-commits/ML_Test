import numpy as np
import tensorflow as tf
import math

# -----------------------------
# SDS PARAMETERS
# -----------------------------
SDS_NUM_MICS = 8
SDS_SAMPLE_RATE = 16000
SDS_FRAME_LEN = 256
c = 343.0

# Microphone positions (same as STM32)
# Replace with your real positions if needed
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
def makeTestFrame360_farfield(angleDeg, distance_m, noise):
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
mel_filterbank = np.load("mel_filterbank_40x129.npy")

def stm32_mel_features(frame):
    import librosa

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
interpreter = tf.lite.Interpreter("models/binary/model_binary_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def infer_binary(feat):
    interpreter.set_tensor(input_details[0]['index'], feat.reshape(1, -1))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return float(out[0])

# -----------------------------
# MAIN TEST LOOP
# -----------------------------
if __name__ == "__main__":
    trueAz = 0.0
    trueDist = 50.0

    dAz = 10.0
    dDist = 5.0

    for i in range(50):
        print(f"\n--- Frame {i} ---")
        print(f"Azimuth: {trueAz:.1f}°  Distance: {trueDist:.1f} m")

        micFrame = makeTestFrame360_farfield(trueAz, trueDist, noise=0.01)

        # Combine microphones (simple sum, like STM32 pre-mix)
        mono = micFrame.mean(axis=0)

        feat = stm32_mel_features(mono)
        pred = infer_binary(feat)

        print(f"Prediction: {pred:.3f}  ->  {'DRONE' if pred > 0.5 else 'NO DRONE'}")

        trueAz += dAz
        if trueAz >= 360.0:
            trueAz = 0.0
            trueDist += dDist
            if trueDist >= 100.0:
                trueDist = 50.0
