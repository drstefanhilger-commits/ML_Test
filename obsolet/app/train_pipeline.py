import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

from app.train.hard_boost import apply_hard_boost

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 4096
MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

# Neue saubere Datenumgebung
DRONE_SRC = Path("data/selected")
DRONE_SIM_SRC = Path("data/Simulation")
NO_DRONE_SRC = Path("data/ESC-50/audio")

DRONE_HARD_SRC = Path("data/train/drone_hard")
NO_DRONE_HARD_SRC = Path("data/train/no_drone_hard")

MODEL_OUT = Path("models/binary/model_binary.h5")
TFLITE_OUT = Path("models/binary/model_binary_int8.tflite")


# ---------------------------------------------------------
# Feature Extraction (STM32-kompatibel)
# ---------------------------------------------------------
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
    mel = MEL_FILTERBANK @ mag
    mel = np.log10(mel + 1e-6)

    return mel.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------
def load_dataset():
    X_paths = []
    Y_labels = []

    def add_folder(folder, label):
        if folder.exists():
            for wav in sorted(folder.glob("*.wav")):
                X_paths.append(wav)
                Y_labels.append(label)

    # Drone
    add_folder(DRONE_SRC, 1)

    # Drone Simulation (class_A ... class_E)
    for cls in DRONE_SIM_SRC.glob("class_*"):
        add_folder(cls, 1)

    # No-Drone (ESC-50)
    add_folder(NO_DRONE_SRC, 0)

    # Hard Samples
    add_folder(DRONE_HARD_SRC, 1)
    add_folder(NO_DRONE_HARD_SRC, 0)

    print(f"Dataset geladen: {len(X_paths)} Samples")
    print(f"  Drohne:   {np.sum(np.array(Y_labels)==1)}")
    print(f"  No-Drone: {np.sum(np.array(Y_labels)==0)}")

    return X_paths, Y_labels


# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------------------------------------------------
# TFLite Export (Int8 Quantization)
# ---------------------------------------------------------
def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data():
        for _ in range(200):
            yield [np.random.rand(1, 40).astype(np.float32)]

    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] TFLite gespeichert: {TFLITE_OUT}")


# ---------------------------------------------------------
# Training Pipeline (mit Hard-Boost)
# ---------------------------------------------------------
def train_pipeline():
    X_paths, Y_labels = load_dataset()

    # Hard-Boost anwenden
    boost_X, boost_Y = apply_hard_boost(X_paths, Y_labels)

    # Features extrahieren
    X = np.array([extract_features(p) for p in X_paths] +
                 [extract_features(p) for p in boost_X], dtype=np.float32)

    Y = np.array(Y_labels + boost_Y, dtype=np.float32)

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    # Train/Test Split
    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Model
    model = build_model()

    # Training
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=20,
        batch_size=32
    )

    # Save Keras model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"[OK] Keras Modell gespeichert: {MODEL_OUT}")

    # Export TFLite
    export_tflite(model)

    # Evaluation
    preds = model.predict(X_test).flatten()
    labels = (preds >= 0.5).astype(int)

    FN = np.sum((labels == 0) & (Y_test == 1))
    FP = np.sum((labels == 1) & (Y_test == 0))

    print("\n=== Evaluation ===")
    print(f"False Negatives: {FN}")
    print(f"False Positives: {FP}")
    print(f"FN-Rate: {FN / np.sum(Y_test==1):.3f}")
    print(f"FP-Rate: {FP / np.sum(Y_test==0):.3f}")

    print(f"Score Durchschnitt: {np.mean(preds):.3f}")
    print(f"Score Minimum:      {np.min(preds):.3f}")
    print(f"Score Maximum:      {np.max(preds):.3f}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    train_pipeline()
