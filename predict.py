import os
import cv2
import numpy as np
import tensorflow as tf

# ===============================
# Load trained CNN model
# ===============================

model = tf.keras.models.load_model("model/mnist_model.keras")


# ===============================
# Image Preprocessing
# ===============================

def preprocess_image(image_path):

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception("Unable to read image.")

    # ---------------------------------
    # Decide whether inversion is needed
    # ---------------------------------

    if np.mean(img) > 127:
        img = 255 - img

    # ---------------------------------
    # Gaussian Blur
    # ---------------------------------

    img = cv2.GaussianBlur(img, (5, 5), 0)

    # ---------------------------------
    # Threshold
    # ---------------------------------

    _, thresh = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---------------------------------
    # Largest contour (digit)
    # ---------------------------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise Exception("Digit not found.")

    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    digit = thresh[y:y+h, x:x+w]

    # ---------------------------------
    # Resize while keeping aspect ratio
    # ---------------------------------

    size = 20

    h, w = digit.shape

    if h > w:

        new_h = size
        new_w = int(w * size / h)

    else:

        new_w = size
        new_h = int(h * size / w)

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------
    # Create 28x28 black canvas
    # ---------------------------------

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    # ---------------------------------
    # Save processed image
    # ---------------------------------

    os.makedirs("static/processed", exist_ok=True)

    cv2.imwrite(
        "static/processed/processed.png",
        canvas
    )

    # ---------------------------------
    # Normalize
    # ---------------------------------

    canvas = canvas.astype("float32") / 255.0

    canvas = canvas.reshape(1, 28, 28, 1)

    return canvas


# ===============================
# Prediction
# ===============================

def predict_digit(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)[0]

    predicted_digit = int(np.argmax(prediction))

    confidence = float(np.max(prediction) * 100)

    top3_indices = prediction.argsort()[-3:][::-1]

    top3 = []

    for index in top3_indices:

        top3.append(
            (
                int(index),
                round(float(prediction[index] * 100), 2)
            )
        )

    return predicted_digit, confidence, top3