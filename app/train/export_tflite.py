import tensorflow as tf

model = tf.keras.models.load_model("models/binary/model_finetuned.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("models/binary/model_finetuned_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported: model_finetuned_int8.tflite")
