import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from tensorflow.keras import layers, models

# ----------------------------------------
# CONFIG
# ----------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 4096

DRONE_DIR = Path("data/train/drone")
DRONE_HARD_DIR = Path("data/train/drone_hard")
NO_DRONE_DIR = Path("data/train/no_drone")

MEL_FILTERBANK = np.load("mel_filterbank_40x129.npy")

# ----------------------------------------
# Feature Extraction
# ----------------------------------------
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

# ----------------------------------------
# Load Dataset
# ----------------------------------------
def load_dataset():
    X = []
    Y = []

    # DRONE = 1
    for p in DRONE_DIR.glob("*.wav"):
        feat = extract_features(p)
        X.append(feat)
        Y.append(1)

    # DRONE_HARD = 1
    for p in DRONE_HARD_DIR.glob("*.wav"):
        feat = extract_features(p)
        X.append(feat)
        Y.append(1)

    # NO DRONE = 0
    for p in NO_DRONE_DIR.glob("*.wav"):
        feat = extract_features(p)
        X.append(feat)
        Y.append(0)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print("Dataset loaded:")
    print("  DRONE:", len([y for y in Y if y == 1]))
    print("  NO DRONE:", len([y for y in Y if y == 0]))
    print("  Total:", len(Y))

    return X, Y

# ----------------------------------------
# Build Model
# ----------------------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(40,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ----------------------------------------
# Train
# ----------------------------------------
def train():
    X, Y = load_dataset()

    model = build_model()

    history = model.fit(
        X, Y,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        shuffle=True
    )

    model.save("models/binary/model_binary_trained.h5")
    print("[OK] Keras-Modell gespeichert.")

    return model

# ----------------------------------------
# Quantize to TFLite INT8
# ----------------------------------------
def quantize(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open("models/binary/model_binary_int8.tflite", "wb") as f:
        f.write(tflite_model)

    print("[OK] INT8 TFLite-Modell gespeichert.")

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    model = train()
    quantize(model)
