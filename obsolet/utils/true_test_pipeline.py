import numpy as np
import librosa
from pathlib import Path
import tensorflow as tf

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

TFLITE_MODEL = Path("models/binary/model_binary_int8.tflite")

TRUE_DIR = Path("test/true_test")        # echte Drohnen-Samples
TRAIN_TRUE_DIR = Path("data/train/drone")  # Trainings-Drohnen

HARD_THRESHOLD = 0.75
NO_DRONE_THRESHOLD = 0.50


# ---------------------------------------------------------
# Feature Extraction (STM32-kompatibel)
# ---------------------------------------------------------
def extract_features(path: Path) -> np.ndarray:
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

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
    mel = MEL_FILTERBANK @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------
# INT8 Quantisierung / Dequantisierung
# ---------------------------------------------------------
def quantize(feat, scale, zero_point):
    return (feat / scale + zero_point).astype(np.int8)


def dequantize(value, scale, zero_point):
    return (value.astype(np.float32) - zero_point) * scale


# ---------------------------------------------------------
# Load INT8 TFLite Model
# ---------------------------------------------------------
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[OK] INT8 TFLite Modell geladen")
    return interpreter, input_details, output_details


# ---------------------------------------------------------
# Inference (INT8 → float Score)
# ---------------------------------------------------------
def infer_score(interpreter, input_details, output_details, feat):
    in_scale, in_zero = input_details[0]['quantization']
    qfeat = quantize(feat, in_scale, in_zero)

    interpreter.set_tensor(input_details[0]['index'], qfeat.reshape(1, -1))
    interpreter.invoke()

    out_raw = interpreter.get_tensor(output_details[0]['index'])[0][0]
    out_scale, out_zero = output_details[0]['quantization']

    score = (out_raw - out_zero) * out_scale
    return float(score)


# ---------------------------------------------------------
# TRUE-TEST PIPELINE
# ---------------------------------------------------------
def collect_true_samples():
    files = []

    if TRAIN_TRUE_DIR.exists():
        files += sorted(TRAIN_TRUE_DIR.glob("*.wav"))

    if TRUE_DIR.exists():
        files += sorted(TRUE_DIR.glob("*.wav"))

    return sorted(set(files))


def run_true_test():
    interpreter, input_details, output_details = load_interpreter()
    files = collect_true_samples()

    print("\n==============================================")
    print(" TRUE-TEST-FIX-PIPELINE (utils)")
    print(" Samples:", len(files))
    print("==============================================\n")

    hard_samples = []
    no_drone_errors = []

    for f in files:
        feat = extract_features(f)
        score = infer_score(interpreter, input_details, output_details, feat)

        label = "DRONE" if score >= NO_DRONE_THRESHOLD else "NO DRONE"

        print(f"=== TRUE-SOUND: {f.name} ===")
        print(f"True-Test -> {score:.3f}  ({label})\n")

        if score < HARD_THRESHOLD:
            hard_samples.append((f, score))

        if label == "NO DRONE":
            no_drone_errors.append((f, score))

    print("\n----------------------------------------------")
    print(" SUMMARY")
    print("----------------------------------------------")
    print("Harte Samples (<0.75):", len(hard_samples))
    print("NO-DRONE Fehler (<0.50):", len(no_drone_errors))
    print("----------------------------------------------\n")

    return hard_samples, no_drone_errors


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    run_true_test()
