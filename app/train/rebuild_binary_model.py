import tensorflow as tf
from tensorflow.keras import layers, models

OLD_MODEL = "data/Existing_ML_data/model.h5"
NEW_MODEL = "models/binary/model_binary_base.h5"

def main():
    old = tf.keras.models.load_model(OLD_MODEL)

    # alle Schichten außer der letzten übernehmen
    x = old.layers[-2].output

    # neue Binary-Schicht
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=old.input, outputs=out)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.save(NEW_MODEL)
    print("Binary model architecture saved:", NEW_MODEL)

if __name__ == "__main__":
    main()
