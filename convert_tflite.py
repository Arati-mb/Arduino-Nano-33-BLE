import tensorflow as tf
import os

# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "tomato_model.keras"
SAVED_MODEL_DIR = "saved_model"
TFLITE_FILE = "tomato_model.tflite"

# ============================================================
# LOAD KERAS MODEL
# ============================================================

print("Loading:", MODEL_FILE)

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Model loaded successfully.")

# ============================================================
# EXPORT AS SAVEDMODEL
# ============================================================

print("\nExporting SavedModel...")

if os.path.exists(SAVED_MODEL_DIR):
    import shutil
    shutil.rmtree(SAVED_MODEL_DIR)

model.export(SAVED_MODEL_DIR)

print("SavedModel created successfully.")

# ============================================================
# CONVERT SAVEDMODEL TO TFLITE
# ============================================================

print("\nConverting SavedModel to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    SAVED_MODEL_DIR
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

# ============================================================
# SAVE TFLITE MODEL
# ============================================================

with open(TFLITE_FILE, "wb") as f:
    f.write(tflite_model)

# ============================================================
# RESULT
# ============================================================

print("\n========================================")
print("TFLITE CONVERSION SUCCESSFUL")
print("========================================")
print("File:", TFLITE_FILE)
print("Size:", len(tflite_model), "bytes")
print("Size:", round(len(tflite_model) / 1024, 2), "KB")
print("========================================")