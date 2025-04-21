import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model("Cat_Dog_Classifier.h5")

# Set the title
st.title("🐱🐶 Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # normalize
    image_array = image_array.reshape((1, 256, 256, 3))  # add batch dimension
    return image_array

# Prediction function
def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_label = "Dog" if prediction[0][0] > 0.5 else "Cat"
    confidence = float(prediction[0][0]) if class_label == "Dog" else 1 - float(prediction[0][0])
    return class_label, confidence

# If image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        label, confidence = classify_image(image)
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence:.2%}")
