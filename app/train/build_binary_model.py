import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Input(shape=(40,)),          # deine STM32-kompatiblen Features
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")   # Binary Output
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    model = build_model()
    model.save("models/binary/model_binary_base.h5")
    print("Binary base model saved.")

if __name__ == "__main__":
    main()
