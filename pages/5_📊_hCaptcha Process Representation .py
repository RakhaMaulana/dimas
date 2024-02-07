import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# Core
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def main():
    st.markdown(
    "<h3 style='text-align: center;'>Dataset Image Captcha</h3>", unsafe_allow_html=True)
    st.markdown(
    "<h6 style='text-align: center;'>Menampilkan sampel 20 gambar dari dataset</h6>", unsafe_allow_html=True)

    dataset_dir = "datasets/hCaptcha_images/test/airplane"

    image_files = os.listdir(dataset_dir)

    col1, col2, col3, col4 = st.columns(4)

    for i in range(20):
        image_path = os.path.join(dataset_dir, image_files[i])
        image = Image.open(image_path)
        
        if i % 4 == 0:
            col1.image(image, caption=f"Gambar ke-{i+1}", use_column_width=True)
        elif i % 4 == 1:
            col2.image(image, caption=f"Gambar ke-{i+1}", use_column_width=True)
        elif i % 4 == 2:
            col3.image(image, caption=f"Gambar ke-{i+1}", use_column_width=True)
        else:
            col4.image(image, caption=f"Gambar ke-{i+1}", use_column_width=True)  
    
    st.markdown(
        "<h3 style='text-align: center;'>Data Augmentation</h3>", unsafe_allow_html=True)
    
    _left, mid, _right = st.columns([0.05, 0.9, 0.05])
    with mid:
        st.image("assets/hCaptcha_Dataset.png", caption="VA", use_column_width=True)
    
    st.markdown(
        "<h3 style='text-align: center;'>Learning Curves</h3>", unsafe_allow_html=True)
    
    _left, mid, _right = st.columns([0.05, 0.9, 0.05])
    with mid:
        st.image("assets/hCaptcha_Validation Accuracy.png", caption="VA", use_column_width=True)
    
    _left, mid, _right = st.columns([0.05, 0.9, 0.05])
    with mid:
        st.image("assets/hcaptcha_Validation Loss.png", caption="VL", use_column_width=True)
    
    st.markdown(
        "<h3 style='text-align: center;'>Evaluate Model</h3>", unsafe_allow_html=True)
    
    _left, mid, _right = st.columns([0.05, 0.9, 0.05])
    with mid:
        st.image("assets/hCaptcha_Confusion.png", caption="C", use_column_width=True)

if __name__ == "__main__":
    main()
