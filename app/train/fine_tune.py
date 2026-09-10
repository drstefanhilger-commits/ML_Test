import tensorflow as tf
import numpy as np
import json
from pathlib import Path

MODEL_PATH = Path("data/Existing_ML_data/model.h5")
SPLIT_PATH = Path("data/splits/extended_train.json")
MEL_DIR = Path("data/mel")

def load_dataset():
    with open(SPLIT_PATH) as f:
        items = json.load(f)

    X, y = [], []

    for item in items:
        mel_path = MEL_DIR / item["mel"]
        feat = np.load(mel_path)
        X.append(feat)
        y.append(item["label"])

    X = np.array(X)
    y = np.array(y)

    return X, y

def main():
    print("Lade Modell…")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Lade Dataset…")
    X, y = load_dataset()
    print(f"Samples: {len(X)}")

    print("Starte Fine‑Tuning…")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X, y,
        batch_size=32,
        epochs=10,
        validation_split=0.1
    )

    OUT = Path("models/binary/model_finetuned.h5")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUT)

    print(f"Fine‑tuned model saved: {OUT}")

if __name__ == "__main__":
    main()
