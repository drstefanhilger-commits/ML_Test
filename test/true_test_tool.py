import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_LEN = 4096

mel_filterbank = np.load("mel_filterbank_40x129.npy")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < FRAME_LEN:
        y = np.pad(y, (0, FRAME_LEN - len(y)))
    else:
        y = y[:FRAME_LEN]

    window = np.hanning(FRAME_LEN)
    S = librosa.stft(
        y,
        n_fft=FRAME_LEN,
        hop_length=FRAME_LEN // 2,
        window=window,
        center=False
    )

    mag = np.abs(S)[:129, :]
    mel = mel_filterbank @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)

# -----------------------------
# Load TFLite Model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="model_binary_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_model(x):
    x = x.reshape(1, 40).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return float(out[0])

# -----------------------------
# True-Test
# -----------------------------
TRUE_DIR = Path("test/true_test")

files = list(TRUE_DIR.glob("*.wav"))
print("Gefundene True-Test WAV-Dateien:", len(files))
print("Starte True-Test Analyse...\n")

for p in files:
    print(f"=== TRUE-SOUND: {p.name} ===")
    feat = extract_features(p)
    score = run_model(feat)
    label = "DRONE" if score >= 0.5 else "NO DRONE"
    print(f"True-Test -> {score:.3f}  ({label})\n")
