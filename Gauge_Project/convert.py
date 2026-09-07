import tensorflow as tf

model = tf.keras.models.load_model("model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")
print("Model size:", len(tflite_model), "bytes")