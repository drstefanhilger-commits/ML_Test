import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
import os

# ----------------------------------------
# 1) Alte Modelle löschen
# ----------------------------------------
for f in ["model_binary.h5", "model_binary_int8.tflite"]:
    if Path(f).exists():
        os.remove(f)
        print("Gelöscht:", f)

# ----------------------------------------
# 2) Audio-Parameter
# ----------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 4096   # WICHTIG: statt 256

mel_filterbank = np.load("mel_filterbank_40x129.npy")

# ----------------------------------------
# 3) Feature-Extraktion
# ----------------------------------------
def extract_features(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    # 256 ms Audio statt 16 ms
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

# ----------------------------------------
# 4) Daten laden
# ----------------------------------------
X = []
Y = []

# Drone
DRONE_DIR = Path("data/train/drone")
for p in DRONE_DIR.glob("*.wav"):
    X.append(extract_features(p))
    Y.append(1)

# No-Drone (beide Ordner!)
NO_DRONE_DIRS = [
    Path("data/train/no_drone"),
    Path("data/train/no_drone_aug")
]

for d in NO_DRONE_DIRS:
    for p in d.glob("*.wav"):
        X.append(extract_features(p))
        Y.append(0)

# Richtige Anzeige
drone_count = len(list(DRONE_DIR.glob("*.wav")))
no_drone_count = sum(len(list(d.glob("*.wav"))) for d in NO_DRONE_DIRS)

print("No-drone samples (real):", sum(len(list(d.glob("*.wav"))) for d in NO_DRONE_DIRS))

print("Drone samples:", drone_count)
print("No-drone samples:", no_drone_count)

X = np.array(X)
Y = np.array(Y)

# ----------------------------------------
# 5) Modell
# ----------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(40,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, Y,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

model.save("model_binary.h5")

# ----------------------------------------
# 6) TFLite Export (INT8)
# ----------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model_binary_int8.tflite", "wb") as f:
    f.write(tflite_model)

size = Path("model_binary_int8.tflite").stat().st_size
print("TFLite model saved: model_binary_int8.tflite")
print("TFLite size:", size, "Bytes")
