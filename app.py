# Simple Streamlit demo app for the chest X-ray classifier
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

from src.model import build_model

WEIGHTS_PATH = "model_weights.h5"

st.title("Chest X-ray Pneumonia Classifier")
st.write("Upload a chest X-ray image to classify it as Pneumonia or Normal.")

model, _ = build_model()

try:
    model.load_weights(WEIGHTS_PATH)
    model_ready = True
except Exception:
    model_ready = False
    st.warning(
        f"No trained weights found at '{WEIGHTS_PATH}'. "
        "Train the model first (see notebooks/analysis.ipynb), then "
        "save it with model.save_weights('model_weights.h5') and rerun this app."
    )

uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model_ready:
        resized = image.resize((224, 224))
        array = np.array(resized).astype("float32")
        array = preprocess_input(array)
        # match the ImageDataGenerator's samplewise_center + samplewise_std_normalization
        # used during training (src/data.py), applied after preprocess_input
        array = array - np.mean(array)
        array = array / (np.std(array) + 1e-6)
        batch = np.expand_dims(array, axis=0)

        prediction = model.predict(batch)[0][0]
        label = "Pneumonia" if prediction >= 0.5 else "Normal"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence score: {prediction:.4f}")
