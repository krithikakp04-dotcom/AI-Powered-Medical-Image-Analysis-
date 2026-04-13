import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Fix random seeds for deterministic inference
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)

# Load model once
model = load_model("medical_model.h5", compile=False)

IMG_SIZE = 224

st.title("🧠 AI Medical Image Analysis System")
st.markdown("Upload a Chest X-ray to detect Pneumonia")


def prepare_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# Upload image
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded Image", use_container_width=True, clamp=True, channels="GRAY")

    # Preprocess
    img_tensor = prepare_image(img)

    # Prediction
    prediction = model.predict(img_tensor, verbose=0)[0][0]

    st.subheader("🧾 Result")

    if prediction > 0.5:
        st.error(f"🧠 Pneumonia Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"🫁 Normal (Confidence: {1 - prediction:.2f})")

# -------------------------------
# SHOW GRAPHS (IMPORTANT ⭐)
# -------------------------------
st.subheader("📊 Model Performance")

if os.path.exists("outputs/accuracy.png"):
    st.image("outputs/accuracy.png")

if os.path.exists("outputs/loss.png"):
    st.image("outputs/loss.png")

if os.path.exists("outputs/confusion_matrix.png"):
    st.image("outputs/confusion_matrix.png")