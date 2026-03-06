import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model\skin_cancer_model.keras") # If shows error - copy the file path of skin_cancer_model.keras and paste it here.

# Class names and symptoms for each class
class_names = ["Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma",
               "Melanoma", "Nevus", "Pigmented Benign Keratosis",
               "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"]

symptoms = {
    "Actinic Keratosis": "Rough, scaly patch on sun-exposed skin.",
    "Basal Cell Carcinoma": "Waxy bump, usually on the face or neck.",
    "Dermatofibroma": "Firm, raised nodule, often red or brown.",
    "Melanoma": "New or unusual growth, often irregular in color and shape.",
    "Nevus": "Common mole, usually benign and small in size.",
    "Pigmented Benign Keratosis": "Non-cancerous skin growth, often brown.",
    "Seborrheic Keratosis": "Wart-like lesion, often dark in color.",
    "Squamous Cell Carcinoma": "Red, firm bump, scaly patch, or sore that heals and reopens.",
    "Vascular Lesion": "Abnormal growth of blood vessels, often appears as red or purple spots."
}

# Function to preprocess the image
def preprocess_image(image, target_size=(180, 180)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# App title and layout with background
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("Skin Cancer Detection App")
st.markdown("This tool uses a Convolutional Neural Network (CNN) to classify images of skin lesions and provides information on possible symptoms.")

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload a clear image of a skin lesion.")
    st.write("2. The model will classify it into one of 9 categories.")
    st.write("3. View the predicted class, confidence level, and description.")
    st.write("---")
    st.write("Developed as part of a semester project.")

# File upload and display option
st.write("### Upload an image for detection")
uploaded_file = st.file_uploader("Choose a skin lesion image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Option to display the uploaded image
    show_image = st.checkbox("Show uploaded image", value=False)
    if show_image:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and make predictions
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][pred_class] * 100
    predicted_class_name = class_names[pred_class]

    # Display prediction and symptoms
    st.write("### Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class_name}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Description:** {symptoms[predicted_class_name]}")

    # Styled prediction display
    st.markdown(f"<div style='text-align: center; font-size: 18px; color: #6c63ff;'>"
                f"Result: {predicted_class_name}</div>", unsafe_allow_html=True)
else:
    st.write("Please upload an image to proceed.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 14px;'>"
            "This app was created for educational purposes as a semester project.</div>",
            unsafe_allow_html=True)
