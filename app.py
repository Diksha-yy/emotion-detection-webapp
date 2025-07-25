import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("emotion_model.h5")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("😊 Emotion Detection from Facial Image")
st.write("Upload an image and get the predicted emotion")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((48, 48))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])

    # Predict
    prediction = model.predict(img_array)
    emotion_label = emotions[np.argmax(prediction)]

    st.subheader("Detected Emotion:")
    st.success(emotion_label)
