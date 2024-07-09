import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
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
st.title('Image Classification with Machine Learning')
st.subheader('By Emuejevoke Eshemitan')
st.image('web_image.jpeg')

st.write('Upload an image to classify it using the trained model.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Classifying...')
    image = preprocess_image(image)

    # Make a prediction
    if model:
        prediction = model.predict(image)
        predicted_class = labels[np.argmax(prediction)]
        # Display the prediction
        st.write(f'Predicted class: {predicted_class}')
    else:
        st.error("Model could not be loaded.")