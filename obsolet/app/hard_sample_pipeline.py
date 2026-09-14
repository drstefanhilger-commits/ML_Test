import shutil
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf

# Pfade
SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

MODEL_PATH = Path("models/binary/model_binary.h5")

TRUE_TEST_DIR = Path("test/true_test")
FALSE_TEST_DIR = Path("test/false_sounds")

DRONE_HARD_DIR = Path("data/train/drone_hard")
NO_DRONE_HARD_DIR = Path("data/train/no_drone_hard")

DRONE_SIM_DIR = Path("data/train/drone_sim")


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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Modell geladen: {MODEL_PATH}")
    return model


def collect_hard_samples(threshold: float = 0.5):
    """
    Sammelt automatisch:
      - False Negatives (echte Drohnen → NO DRONE)
      - False Positives (Umwelt → DRONE)
    und kopiert sie in:
      - data/train/drone_hard/
      - data/train/no_drone_hard/
    """

    DRONE_HARD_DIR.mkdir(parents=True, exist_ok=True)
    NO_DRONE_HARD_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    fn_count = 0
    fp_count = 0

    # --- False Negatives (echte Drohnen, aber Score < threshold) ---
    for wav in sorted(TRUE_TEST_DIR.glob("*.wav")):
        feat = extract_features(wav)
        pred = float(model.predict(feat.reshape(1, -1))[0][0])

        if pred < threshold:
            shutil.copy(wav, DRONE_HARD_DIR / wav.name)
            fn_count += 1
            print(f"[FN] {wav.name}  ->  {pred:.3f}  (kopiert nach drone_hard)")

    # --- False Positives (Umwelt, aber Score >= threshold) ---
    for wav in sorted(FALSE_TEST_DIR.glob("*.wav")):
        feat = extract_features(wav)
        pred = float(model.predict(feat.reshape(1, -1))[0][0])

        if pred >= threshold:
            shutil.copy(wav, NO_DRONE_HARD_DIR / wav.name)
            fp_count += 1
            print(f"[FP] {wav.name}  ->  {pred:.3f}  (kopiert nach no_drone_hard)")

    print("\n=== Hard-Sample Zusammenfassung ===")
    print(f"Neue False Negatives (FN): {fn_count}")
    print(f"Neue False Positives (FP): {fp_count}")


def reduce_simulations(max_count: int = 400):
    """
    Reduziert die Anzahl der Simulationen, um FP-Bias zu verringern.
    Behält nur max_count Dateien in data/train/drone_sim/.
    """

    if not DRONE_SIM_DIR.exists():
        print("[INFO] Kein drone_sim Ordner vorhanden.")
        return

    sims = sorted(DRONE_SIM_DIR.glob("*.wav"))
    total = len(sims)

    if total <= max_count:
        print(f"[OK] Simulationen bereits <= {max_count} (aktuell {total})")
        return

    # Dateien, die behalten werden
    keep = sims[:max_count]

    # Rest löschen
    for wav in sims[max_count:]:
        wav.unlink()

    print(f"[OK] Simulationen reduziert: {total} -> {len(keep)}")


def hard_sample_pipeline(threshold: float = 0.5, max_sim_count: int = 400):
    """
    Vollständige Hard-Sample-Pipeline:
      1) Hard-Samples (FN/FP) sammeln
      2) Simulationen reduzieren
      3) Danach manuell: train_pipeline() erneut ausführen
    """

    print("\n=== HARD-SAMPLE PIPELINE START ===")

    collect_hard_samples(threshold=threshold)
    reduce_simulations(max_count=max_sim_count)

    print("\n=== HARD-SAMPLE PIPELINE DONE ===")
    print("Bitte jetzt: train_pipeline() erneut ausführen.")


if __name__ == "__main__":
    # Standard: Threshold 0.5, max 400 Simulationen
    hard_sample_pipeline(threshold=0.5, max_sim_count=400)
