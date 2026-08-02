import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# This assumes your model is in the same folder as app.py
@st.cache_resource
def load_trained_model():
    # You will upload the .keras file to your GitHub along with this script
    return load_model("brain_tumor_model.keras")

model = load_trained_model()
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.title("Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI", use_container_width=True)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button("Predict"):
        prediction = model.predict(img_array)
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")
