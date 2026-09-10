# Dronen notebook – Drone + ESC-50 Training für STM32

# ============================================================
# 0. Setup
# ============================================================

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Projektpfade (bitte anpassen)
DRONE_ROOT = r"./DroneAudioDataset"      # Klon von https://github.com/saraalemadi/DroneAudioDataset
ESC50_ROOT = r"./ESC-50"                 # Pfad zu ESC-50 (CSV + audio)
SR = 16000                               # Ziel-Samplingrate
N_MELS = 40                              # Mel-Features (STM32-kompatibel)
FFT_SIZE = 512                           # Beispiel-FFT-Größe
HOP_LENGTH = 256                         # Hop-Länge

# ============================================================
# 1. Hilfsfunktionen: Audio laden + Mel-Features
# ============================================================

def load_audio(path, sr=SR):
    y, s = librosa.load(path, sr=sr, mono=True)
    return y, s

def extract_mel_features(y, sr=SR, n_mels=N_MELS, fft_size=FFT_SIZE, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=fft_size,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Für STM32: 1 Vektor aus 40 Werten → Mittelwert über Zeit
    feat = S_db.mean(axis=1)
    return feat.astype(np.float32)

# ============================================================
# 2. DroneAudioDataset laden und 200 Samples selektieren
# ============================================================

def collect_drone_files(root, max_files=200):
    drone_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith(".wav"):
                continue
            path = os.path.join(dirpath, f)
            drone_files.append(path)
    # einfache Auswahl: erste 200, später verfeinern
    drone_files = drone_files[:max_files]
    return drone_files

drone_files = collect_drone_files(DRONE_ROOT, max_files=200)
print("Gefundene Drone-Dateien:", len(drone_files))

# ============================================================
# 3. ESC-50 laden (Human / Wind / Background)
# ============================================================

import pandas as pd

ESC50_META = os.path.join(ESC50_ROOT, "meta", "esc50.csv")
df = pd.read_csv(ESC50_META)

# Mapping: ESC-50 Labels → unsere Klassen
ESC_LABEL_MAP = {
    "speech": "human",
    "crowd": "human",
    "wind": "wind",
    "rain": "background",
    "traffic": "background",
    "car_horn": "background",
    "engine": "background",
    # hier kannst du erweitern
}

def collect_esc_files(root, df, label_map):
    files = []
    labels = []
    for _, row in df.iterrows():
        esc_label = row["category"]
        if esc_label not in label_map:
            continue
        cls = label_map[esc_label]
        path = os.path.join(root, "audio", row["filename"])
        files.append(path)
        labels.append(cls)
    return files, labels

esc_files, esc_labels = collect_esc_files(ESC50_ROOT, df, ESC_LABEL_MAP)
print("ESC-50 Dateien:", len(esc_files))

# ============================================================
# 4. Features + Labels bauen (Drone + ESC-50)
# ============================================================

X = []
y = []

# Klassen-Index
CLASS_MAP = {
    "drone": 0,
    "human": 1,
    "wind": 2,
    "background": 3,
}

# 4.1 Drone-Features
for path in drone_files:
    try:
        audio, sr = load_audio(path)
        if len(audio) < sr:  # mindestens 1 Sekunde
            continue
        feat = extract_mel_features(audio, sr)
        X.append(feat)
        y.append(CLASS_MAP["drone"])
    except Exception as e:
        print("Fehler Drone:", path, e)

# 4.2 ESC-Features
for path, lbl in zip(esc_files, esc_labels):
    try:
        audio, sr = load_audio(path)
        if len(audio) < sr:
            continue
        feat = extract_mel_features(audio, sr)
        X.append(feat)
        y.append(CLASS_MAP[lbl])
    except Exception as e:
        print("Fehler ESC:", path, e)

X = np.stack(X)
y = np.array(y)

print("Gesamt-Features:", X.shape, "Labels:", y.shape)

# One-Hot
y_oh = tf.keras.utils.to_categorical(y, num_classes=4)

# ============================================================
# 5. Train/Test-Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_oh, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# ============================================================
# 6. Modell definieren (kleines Dense-Netz)
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_MELS,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 7. Training
# ============================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.title("Accuracy")
plt.show()

# ============================================================
# 8. Evaluation
# ============================================================

y_pred = model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_cls, y_pred_cls)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["drone", "human", "wind", "background"]
)
disp.plot(cmap="Blues")
plt.show()

# ============================================================
# 9. TFLite-Konvertierung
# ============================================================

# 9.1 Float32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("drone_model_float32.tflite", "wb") as f:
    f.write(tflite_model)

# 9.2 Int8 quantisiert
def representative_dataset():
    for i in range(100):
        idx = np.random.randint(0, X_train.shape[0])
        x = X_train[idx:idx+1]
        yield [x.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_types = [tf.float16]
tflite_int8 = converter.convert()

with open("drone_model_int8.tflite", "wb") as f:
    f.write(tflite_int8)

print("TFLite-Modelle gespeichert.")

# ============================================================
# 10. Export für STM32 (model_data.cc)
# ============================================================

def tflite_to_c_array(tflite_path, var_name):
    with open(tflite_path, "rb") as f:
        data = f.read()
    hex_array = ", ".join(f"0x{b:02x}" for b in data)
    c_code = f"""
#include <stdint.h>

const unsigned char {var_name}[] = {{
    {hex_array}
}};

const unsigned int {var_name}_len = {len(data)};
"""
    return c_code

c_code = tflite_to_c_array("drone_model_int8.tflite", "drone_model_tflite")
with open("model_data.cc", "w") as f:
    f.write(c_code)

print("model_data.cc erzeugt.")
