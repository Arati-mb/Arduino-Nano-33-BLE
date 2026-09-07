import os
import json
import cv2
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

JSON_PATH = "dataset/labels.json"
IMAGE_SIZE = 96

print("========================================")
print("      SMALL GAUGE MODEL TRAINING")
print("========================================")

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print("Total Records:", len(data))

X = []
labels = []

for item in data:
    image_path = os.path.join("dataset", item["image"])

    image = cv2.imread(image_path)

    if image is None:
        print("Warning: Could not load:", image_path)
        continue

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype(np.float32) / 255.0

    X.append(image)
    labels.append(item["pressure"])

X = np.array(X, dtype=np.float32)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

class_names = label_encoder.classes_

print()
print("Classes:")
print(class_names)

print()
print("Class count:")
print(Counter(y))

with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

print()
print("Creating small CNN...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(96, 96, 3)),

    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print()
print("========================================")
print("           TRAINING MODEL")
print("========================================")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print()
print("========================================")
print("           MODEL EVALUATION")
print("========================================")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

model.save("model_small.keras")

print()
print("Saved: model_small.keras")

print()
print("========================================")
print("       CONVERTING TO INT8 TFLITE")
print("========================================")

def representative_dataset():
    count = min(100, len(X_train))

    for i in range(count):
        sample = X_train[i:i+1]
        yield [sample.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print()
print("Saved: model_quantized.tflite")
print("TFLite size:", len(tflite_model), "bytes")
print("TFLite size:", len(tflite_model) / 1024, "KB")

print()
print("========================================")
print("          CREATING model.h")
print("========================================")

with open("model.h", "w") as f:

    f.write("#ifndef MODEL_H\n")
    f.write("#define MODEL_H\n\n")

    f.write("const unsigned char model[] = {\n")

    for i, byte in enumerate(tflite_model):

        if i % 12 == 0:
            f.write("    ")

        f.write("0x{:02x}".format(byte))

        if i != len(tflite_model) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    if len(tflite_model) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")

    f.write(
        "const unsigned int model_len = "
        + str(len(tflite_model))
        + ";\n\n"
    )

    f.write("#endif\n")

print("Saved: model.h")

print()
print("========================================")
print("              COMPLETE")
print("========================================")

print()
print("Final files:")
print("model_small.keras")
print("model_quantized.tflite")
print("model.h")
print("labels.txt")

print()
print("Model size:", len(tflite_model) / 1024, "KB")