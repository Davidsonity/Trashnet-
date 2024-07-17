import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained EfficientNetB0 model
try:
    model = load_model('efficientnet_model.h5', compile=False)
except TypeError as e:
    st.error(f"Error loading model: {e}")

# Define the labels (replace with your actual labels)
labels = ['metal', 'glass', 'paper', 'trash', 'plastic']

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app
st.title('Solid Waste Classification Using Deep Learning')
st.subheader('By Emuejevoke Eshemitan')

# Display the university logo

st.image('image1.png', caption='University of Lagos', use_column_width=True)

st.info('Please use images with a white background for better accuracy as the model was trained on images with a white background.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

# Camera input
camera_image = st.camera_input('Take a picture')

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
elif camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption='Captured Image', use_column_width=True)

if image is not None:
    image = preprocess_image(image)

    # Make a prediction
    with st.spinner('Classifying...'):
        if model:
            prediction = model.predict(image)
            predicted_class = labels[np.argmax(prediction)]
            # Display the prediction
            st.success(f'Predicted class: {predicted_class}')
        else:
            st.error("Model could not be loaded.")
