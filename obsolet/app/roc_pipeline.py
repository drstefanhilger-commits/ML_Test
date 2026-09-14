import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Pfade
SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

MODEL_PATH = Path("models/binary/model_binary.h5")
THRESHOLD_OUT = Path("models/binary/optimal_threshold.txt")

TRUE_TEST_DIR = Path("test/true_test")
FALSE_TEST_DIR = Path("test/false_sounds")


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
# ROC Pipeline
# ---------------------------------------------------------
def roc_pipeline():
    print("\n=== ROC-PIPELINE START ===")

    # Modell laden
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Modell geladen: {MODEL_PATH}")

    scores = []
    labels = []

    # Drohnen (Label 1)
    for wav in sorted(TRUE_TEST_DIR.glob("*.wav")):
        feat = extract_features(wav)
        pred = float(model.predict(feat.reshape(1, -1))[0][0])
        scores.append(pred)
        labels.append(1)

    # No-Drone (Label 0)
    for wav in sorted(FALSE_TEST_DIR.glob("*.wav")):
        feat = extract_features(wav)
        pred = float(model.predict(feat.reshape(1, -1))[0][0])
        scores.append(pred)
        labels.append(0)

    scores = np.array(scores)
    labels = np.array(labels)

    # ROC berechnen
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Optimaler Threshold (Youden-Index)
    youden = tpr - fpr
    idx = np.argmax(youden)
    optimal_threshold = thresholds[idx]

    print(f"\n=== ROC-Ergebnis ===")
    print(f"AUC: {roc_auc:.3f}")
    print(f"Optimaler Threshold (Youden): {optimal_threshold:.3f}")

    # Threshold speichern
    THRESHOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_OUT, "w") as f:
        f.write(str(optimal_threshold))

    print(f"[OK] Threshold gespeichert in: {THRESHOLD_OUT}")

    # FN/FP bei optimalem Threshold berechnen
    preds = (scores >= optimal_threshold).astype(int)

    FN = np.sum((preds == 0) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))

    print("\n=== Evaluation mit optimalem Threshold ===")
    print(f"False Negatives: {FN}")
    print(f"False Positives: {FP}")
    print(f"FN-Rate: {FN / np.sum(labels == 1):.3f}")
    print(f"FP-Rate: {FP / np.sum(labels == 0):.3f}")

    print("\n=== ROC-PIPELINE DONE ===")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    roc_pipeline()
