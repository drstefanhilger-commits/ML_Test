import numpy as np
import librosa
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

TFLITE_MODEL = Path("models/binary/model_binary_int8.tflite")
TRUE_TEST_DIR = Path("test/true_test")
FALSE_TEST_DIR = Path("test/false_sounds")

THRESHOLD_OUT = Path("models/binary/optimal_threshold_int8.txt")


# ---------------------------------------------------------
# Feature Extraction (STM32-kompatibel)
# ---------------------------------------------------------
def extract_features(path: Path) -> np.ndarray:
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
    mel = MEL_FILTERBANK @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------
# INT8 Quantisierung / Dequantisierung
# ---------------------------------------------------------
def quantize(feat: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (feat / scale + zero_point).astype(np.int8)


def dequantize(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (value.astype(np.float32) - zero_point) * scale


# ---------------------------------------------------------
# Load TFLite INT8 Model
# ---------------------------------------------------------
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[OK] INT8 TFLite Modell geladen")
    print("Input:", input_details)
    print("Output:", output_details)

    return interpreter, input_details, output_details


# ---------------------------------------------------------
# Inference (INT8 → float Score)
# ---------------------------------------------------------
def infer_score(interpreter, input_details, output_details, feat: np.ndarray) -> float:
    in_scale, in_zero = input_details[0]['quantization']
    qfeat = quantize(feat, in_scale, in_zero)

    interpreter.set_tensor(input_details[0]['index'], qfeat.reshape(1, -1))
    interpreter.invoke()

    out_raw = interpreter.get_tensor(output_details[0]['index'])[0][0]
    out_scale, out_zero = output_details[0]['quantization']

    score = (out_raw - out_zero) * out_scale
    return float(score)


# ---------------------------------------------------------
# ROC-Pipeline INT8
# ---------------------------------------------------------
def roc_pipeline_int8():
    print("\n=== INT8 ROC-PIPELINE START ===")

    interpreter, input_details, output_details = load_interpreter()

    scores = []
    labels = []

    # Drohnen (Label 1)
    for wav in sorted(TRUE_TEST_DIR.glob("*.wav")):
        feat = extract_features(wav)
        s = infer_score(interpreter, input_details, output_details, feat)
        scores.append(s)
        labels.append(1)

    # No-Drone (Label 0)
    for wav in sorted(FALSE_TEST_DIR.glob("*.wav")):
        feat = extract_features(wav)
        s = infer_score(interpreter, input_details, output_details, feat)
        scores.append(s)
        labels.append(0)

    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Sicherheitscheck
    if np.all(scores == scores[0]):
        print("\n[WARN] Alle Scores sind identisch – ROC nicht sinnvoll.")
        print("Min:", float(scores.min()), "Max:", float(scores.max()))
        return

    # ROC berechnen
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    youden = tpr - fpr
    idx = np.argmax(youden)
    optimal_threshold = thresholds[idx]

    print("\n=== ROC-Ergebnis (INT8) ===")
    print(f"AUC: {roc_auc:.3f}")
    print(f"Optimaler Threshold (Youden): {optimal_threshold:.3f}")

    THRESHOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_OUT, "w") as f:
        f.write(str(optimal_threshold))

    print(f"[OK] Threshold gespeichert in: {THRESHOLD_OUT}")

    # Evaluation mit optimalem Threshold
    preds = (scores >= optimal_threshold).astype(int)

    FN = np.sum((preds == 0) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    fn_rate = FN / n_pos if n_pos > 0 else 0.0
    fp_rate = FP / n_neg if n_neg > 0 else 0.0

    print("\n=== Evaluation mit optimalem Threshold (INT8) ===")
    print(f"False Negatives: {FN}")
    print(f"False Positives: {FP}")
    print(f"FN-Rate: {fn_rate:.3f}")
    print(f"FP-Rate: {fp_rate:.3f}")

    print("\n=== INT8 ROC-PIPELINE DONE ===")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    roc_pipeline_int8()
