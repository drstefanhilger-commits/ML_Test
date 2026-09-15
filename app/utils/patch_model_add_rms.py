import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "models/v7/model_binary_fp32_v7.h5"
NEW_MODEL_PATH = "models/v7/model_binary_fp32_v7_rms.h5"

# altes Modell laden
old_model = keras.models.load_model(MODEL_PATH)

# Inputs definieren
input_feat = keras.Input(shape=(10240,), name="feat")
input_rms  = keras.Input(shape=(1,), name="rms")

# beide Inputs zusammenführen
x = keras.layers.Concatenate()([input_feat, input_rms])

# alte Modellstruktur weiterverwenden
output = old_model(x)

# neues Modell erzeugen
new_model = keras.Model(inputs=[input_feat, input_rms], outputs=output)

# speichern
new_model.save(NEW_MODEL_PATH)

print("Neues Modell gespeichert:", NEW_MODEL_PATH)
