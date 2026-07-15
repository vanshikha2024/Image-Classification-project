import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =====================================
# Create model folder
# =====================================

os.makedirs("model", exist_ok=True)

# =====================================
# Load Dataset
# =====================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# =====================================
# Normalize
# =====================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# =====================================
# Reshape
# =====================================

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# =====================================
# Data Augmentation
# =====================================

datagen = ImageDataGenerator(
    rotation_range=12,
    zoom_range=0.12,
    width_shift_range=0.12,
    height_shift_range=0.12
)

datagen.fit(x_train)

# =====================================
# Build CNN
# =====================================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same",
        input_shape=(28,28,1)
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(
        64,
        (3,3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3,3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(),
    Dropout(0.25),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )

])

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

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/mnist_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =====================================
# Train
# =====================================

history = model.fit(

    datagen.flow(
        x_train,
        y_train,
        batch_size=64
    ),

    epochs=20,

    validation_data=(x_test, y_test),

    callbacks=[
        early_stop,
        reduce_lr,
        checkpoint
    ],

    verbose=1

)

# =====================================
# Evaluate
# =====================================

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n" + "="*45)
print(f" Test Accuracy : {accuracy*100:.2f}%")
print("="*45)

# =====================================
# Save Final Model
# =====================================

model.save("model/mnist_model.keras")

print("\n✅ CNN Model Saved Successfully!")

# =====================================
# Plot Accuracy
# =====================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

# =====================================
# Plot Loss
# =====================================

plt.subplot(1,2,2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.show()