import tensorflow as tf
import os
import shutil

MODEL_FILE = "tomato_model.keras"
SAVED_MODEL_DIR = "saved_model_int8"
OUTPUT_FILE = "tomato_model_int8.tflite"

IMG_SIZE = 64


# ==========================================
# Load Model
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Model loaded successfully.")


# ==========================================
# Export SavedModel
# ==========================================

print("\nExporting SavedModel...")

if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

model.export(SAVED_MODEL_DIR)

print("SavedModel created successfully.")


# ==========================================
# Representative Dataset
# ==========================================

def representative_dataset():

    dataset = tf.keras.utils.image_dataset_from_directory(
        "dataset",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=True
    )

    for images, labels in dataset.take(100):

        images = tf.cast(images, tf.float32)

        yield [images]


# ==========================================
# INT8 Conversion
# ==========================================

print("\nStarting INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    SAVED_MODEL_DIR
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


print("Converting to INT8...")

tflite_model = converter.convert()


# ==========================================
# Save Model
# ==========================================

with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)


# ==========================================
# Result
# ==========================================

print("\n========================================")
print("INT8 QUANTIZATION SUCCESSFUL")
print("========================================")
print("File:", OUTPUT_FILE)
print("Size:", len(tflite_model), "bytes")
print("Size:", round(len(tflite_model) / 1024, 2), "KB")
print("========================================")