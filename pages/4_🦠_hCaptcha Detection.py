import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('models/model.h5')

# Classes
classes = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck']

# Function to preprocess image
def preprocess_image(image_data):
    img = image.load_img(image_data, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Main function
def main():
    
    st.markdown(
    "<h1 style='text-align: center;'>Image hCaptcha Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown(
    "<h4 style='text-align: center;'>Masukkan Image hCaptcha Untuk Diproses</h4>", unsafe_allow_html=True)
    
    st.markdown(
    "<h6 style='text-align: center;'>[airplane, bicycle, boat, motorbus, motorcycle, seaplane, train, truck]</h6>", unsafe_allow_html=True)
    
    st.markdown(
    "<h4 style='text-align: center;'>Contoh :</h4>", unsafe_allow_html=True)
    
    _left, mid, _right = st.columns([0.2, 0.6, 0.2])
    with mid:
        st.image("datasets/hCaptcha_images/test/airplane/1650199961986_9.jpg", caption="Model Image", use_column_width=True)
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Preprocess image
        image_array = preprocess_image(uploaded_file)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = classes[np.argmax(prediction)]

        # Display prediction
        st.write(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main()
