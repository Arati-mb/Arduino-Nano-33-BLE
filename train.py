import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# =====================================
# Settings
# =====================================

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30

# =====================================
# Load Dataset
# =====================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("\nClasses:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# =====================================
# Model
# =====================================

model = Sequential([

    tf.keras.Input(shape=(64, 64, 3)),

    layers.Rescaling(1./255),

    layers.Conv2D(8, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(24, 3, padding="same", activation="relu"),

    layers.GlobalAveragePooling2D(),

    layers.Dense(2, activation="softmax")
])

# =====================================
# Summary
# =====================================

model.summary()

# =====================================
# Compile
# =====================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# Callbacks
# =====================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# =====================================
# Train
# =====================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# =====================================
# Evaluate
# =====================================

loss, accuracy = model.evaluate(val_ds)

print("\n==============================")
print("Validation Accuracy:", accuracy)
print("==============================")

# =====================================
# Save Model
# =====================================

model.save("tomato_model.keras")

print("\nModel saved as tomato_model.keras")

# =====================================
# Plot Accuracy
# =====================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")

plt.legend()
plt.grid(True)

plt.show()