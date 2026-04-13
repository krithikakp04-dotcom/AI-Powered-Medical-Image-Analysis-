import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Fix random seeds for deterministic inference
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
model = load_model("medical_model.h5", compile=False)

IMG_SIZE = 224


def prepare_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_image(image_path):

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Image not found!")
        return

    # Resize and normalize
    img = prepare_image(img)

    # Prediction
    prediction = model.predict(img, verbose=0)[0][0]

    # Output result
    if prediction > 0.5:
        print("🧠 RESULT: PNEUMONIA DETECTED")
        print(f"Confidence: {prediction:.2f}")
    else:
        print("🫁 RESULT: NORMAL")
        print(f"Confidence: {1 - prediction:.2f}")

# -------------------------------
# TEST IMAGE (CHANGE PATH)
# -------------------------------
predict_image("test/test.jpg")