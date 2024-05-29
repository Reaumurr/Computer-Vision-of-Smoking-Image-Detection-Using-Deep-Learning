import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array
import os


@st.cache_resource
def load_model():
    model_path = "model.keras"  # Update with the absolute file path
    return tf.keras.models.load_model(model_path)

model = load_model()

def prepare_image(img):
    img = img.resize((220, 220))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = "Smoking" if prediction > 0.5 else "Not Smoking"
    
    return predicted_class, prediction[0]

def run():
    st.title("Smoking or Not Smoking Detection")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        
        # Create the directory if it doesn't exist
        upload_dir = './upload_images/'
        os.makedirs(upload_dir, exist_ok=True)
        
        save_image_path = os.path.join(upload_dir, img_file.name)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        predicted_class, score = prepare_image(img)
        st.success(f"**Predicted : {predicted_class}, Score: {score}**")

run()
