import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.configs import BaseModelConfigs

st.set_page_config(
    page_title="Captcha Detector",
    page_icon="assets/logo.png",
    layout="centered"
)

def t_img (img) :
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

def c_img (img) :
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))

def d_img (img) :
    return cv2.dilate(img, np.ones((2,2), np.uint8), iterations = 1)

def b_img (img) :
    return cv2.GaussianBlur(img, (1,1), 0)

path = 'Datasets/captcha_images_v2/samples/'

import os
from PIL import Image
from keras.preprocessing.image import img_to_array, ImageDataGenerator

X = []
y = []

for image in os.listdir(path) :

    if image[6:] != 'png' :
        continue

    img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

    img = t_img(img)
    img = c_img(img)
    img = d_img(img)
    img = b_img(img)

    image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]

    for i in range(5) :
        X.append(img_to_array(Image.fromarray(image_list[i])))
        y.append(image[i])
X = np.array(X)
y = np.array(y)

X /= 255.0

temp = set(y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

y_combine = LabelEncoder().fit_transform(y)
y_one_hot = OneHotEncoder(sparse_output = False).fit_transform(y_combine.reshape(len(y_combine),1))

info = {y_combine[i] : y[i] for i in range(len(y))}

model = load_model('models/result_model.h5')

def get_demo(uploaded_img):
    # Convert uploaded image to numpy array
    img_np = np.array(uploaded_img)

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(img_gray, 'gray')
    plt.axis('off')
    plt.show()
    
    img = t_img(img_gray)
    img = c_img(img)
    img = d_img(img)
    img = b_img(img)
    
    image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
    
    plt.imshow(img, 'gray')
    plt.axis('off')
    
    Xdemo = []
    for i in range(5) :
        Xdemo.append(img_to_array(Image.fromarray(image_list[i])))
    
    Xdemo = np.array(Xdemo)
    Xdemo/= 255.0
    
    ydemo = model.predict(Xdemo)
    ydemo = np.argmax(ydemo, axis = 1)
    
    predicted_chars = ''.join([info[res] for res in ydemo])
    spaced_chars = ' '.join(predicted_chars)
    
    st.markdown(f"<h6 style='text-align: center;'>Predictions:</h6>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>{spaced_chars}</h4>", unsafe_allow_html=True)
    
def main():
    
    st.markdown(
    "<h1 style='text-align: center;'>Text Captcha Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown(
    "<h4 style='text-align: center;'>Masukkan Text Image Captcha Untuk Diproses</h4>", unsafe_allow_html=True)
    
    st.markdown(
    "<h4 style='text-align: center;'>Contoh :</h4>", unsafe_allow_html=True)
    
    _left, mid, _right = st.columns([0.2, 0.6, 0.2])
    with mid:
        st.image("datasets/captcha_images_v2/2g783.png", caption="Model Image", use_column_width=True)
    
    uploaded_file = st.file_uploader("Upload Image :", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:

        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        get_demo(img)
        
if __name__ == '__main__':
    main()
