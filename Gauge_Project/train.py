import os
import json
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

# ----------------------------------------
# Read JSON File
# ----------------------------------------

json_path = "dataset/labels.json"

print("JSON Path:", os.path.abspath(json_path))

with open(json_path, "r") as file:
    labels = json.load(file)

print("Total Records:", len(labels))
print("First Record:", labels[0])

# ----------------------------------------
# Create Empty Lists
# ----------------------------------------

images = []
pressure_labels = []
image_paths = []

# ----------------------------------------
# Read Every Image
# ----------------------------------------

for item in labels:

    image_path = os.path.join("dataset", item["image"])

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        continue

    image = cv2.resize(image, (96, 96))

    images.append(image)
    pressure_labels.append(item["pressure"])
    image_paths.append(item["image"])

# ----------------------------------------
# Convert to NumPy Arrays
# ----------------------------------------

images = np.array(images)
pressure_labels = np.array(pressure_labels)

# ----------------------------------------
# Normalize Images
# ----------------------------------------

images = images.astype("float32") / 255.0

# ----------------------------------------
# Encode Labels
# ----------------------------------------

label_encoder = LabelEncoder()

pressure_labels = label_encoder.fit_transform(pressure_labels)

print("\nEncoded Labels:")
print(pressure_labels[:20])

print("\nPressure Classes:")
print(label_encoder.classes_)

# ----------------------------------------
# Count Images Per Class
# ----------------------------------------

print("\nLabel Counts:")
print(Counter(pressure_labels))

# ----------------------------------------
# Split Dataset
# ----------------------------------------

X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
    images,
    pressure_labels,
    image_paths,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------
# Build CNN
# ----------------------------------------

model = Sequential([

    Conv2D(32, (3,3), activation='relu',
           input_shape=(96,96,3)),

    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(len(label_encoder.classes_), activation='softmax')

])

# ----------------------------------------
# Compile CNN
# ----------------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------
# Show Model Summary
# ----------------------------------------

model.summary()

# ----------------------------------------
# Train CNN
# ----------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ----------------------------------------
# Evaluate CNN
# ----------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\n========== MODEL EVALUATION ==========")
print("Test Accuracy :", test_accuracy)
print("Test Loss     :", test_loss)

# ----------------------------------------
# Predict Test Images
# ----------------------------------------

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

correct = 0
wrong = 0

print("\n========== DETAILED EVALUATION ==========\n")

for i in range(len(y_test)):

    actual = label_encoder.inverse_transform([y_test[i]])[0]
    predicted = label_encoder.inverse_transform([predicted_labels[i]])[0]

    # Actual Status
    if actual in ["0-20", "0-50"]:
        actual_status = "UNSAFE (Low Pressure)"
    elif actual in ["81-100", "201-250", "251-300"]:
        actual_status = "UNSAFE (High Pressure)"
    else:
        actual_status = "SAFE"

    # Predicted Status
    if predicted in ["0-20", "0-50"]:
        predicted_status = "UNSAFE (Low Pressure)"
    elif predicted in ["81-100", "201-250", "251-300"]:
        predicted_status = "UNSAFE (High Pressure)"
    else:
        predicted_status = "SAFE"

    if predicted_labels[i] == y_test[i]:

        correct += 1

        print("CORRECT")
        print("Image              :", test_paths[i])
        print("Actual Pressure    :", actual)
        print("Predicted Pressure :", predicted)
        print("Actual Status      :", actual_status)
        print("Predicted Status   :", predicted_status)
        print("---------------------------------------")

    else:

        wrong += 1

        print("WRONG")
        print("Image              :", test_paths[i])
        print("Actual Pressure    :", actual)
        print("Predicted Pressure :", predicted)
        print("Actual Status      :", actual_status)
        print("Predicted Status   :", predicted_status)
        print("---------------------------------------")

# ----------------------------------------
# Final Summary
# ----------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Total Test Images :", len(y_test))
print("Correct Images    :", correct)
print("Wrong Images      :", wrong)
print("Accuracy          : {:.2f}%".format((correct / len(y_test)) * 100))

# ----------------------------------------
# Save Model
# ----------------------------------------

model.save("model.keras")

print("\nModel saved successfully!")

# ----------------------------------------
# Dataset Information
# ----------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("Images Loaded :", len(images))
print("Labels Loaded :", len(pressure_labels))
print("Image Shape   :", images.shape)
print("Label Shape   :", pressure_labels.shape)

print("\n========== DATASET SPLIT ==========")

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)
print("Training Labels :", y_train.shape)
print("Testing Labels  :", y_test.shape)

print("========================================")