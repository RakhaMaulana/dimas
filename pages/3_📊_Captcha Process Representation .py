import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.preprocessing.image import img_to_array, ImageDataGenerator 
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

st.set_page_config(
    page_title="Captcha Detector",
    page_icon="assets/logo.png",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>Process Representation</h1>", unsafe_allow_html=True)

st.markdown(
    "<h8 style='text-align: justify;'>Cara kerja machine learning sebenarnya berbeda-beda sesuai dengan teknik atau metode pembelajaran seperti apa yang kamu gunakan pada ML. Namun pada dasarnya prinsip cara kerja pembelajaran mesin masih sama, meliputi pengumpulan data, eksplorasi data, pemilihan model atau teknik, memberikan pelatihan terhadap model yang dipilih dan mengevaluasi hasil dari ML.</h8>", unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Captcha Dataset","Image Processing", "Label Distribution", "Data Augmentation", "Model Evaluation"])

with tab1:
   
    def show_top_20_images():
        dataset_path = "datasets/captcha_images_v2"
        files = os.listdir(dataset_path)[:20]
        num_rows = len(files) // 4 + (1 if len(files) % 4 != 0 else 0)

        for i in range(num_rows):
            row_images = files[i * 4: (i + 1) * 4]
            row = st.columns(4)
            for j, file_name in enumerate(row_images):
                file_path = os.path.join(dataset_path, file_name)
                label = file_name.split('_')[0] 
                image = Image.open(file_path)
                label = label.split('.')[0]
                row[j].write(f"Label: {label}")
                row[j].image(image, caption=file_name, use_column_width=True)
    def main():
        
        st.markdown(
            "<h4 style='text-align: center;'>Dataset Captcha</h4>", unsafe_allow_html=True)
        st.write("Setelah menginstal paket-paket yang diperlukan, kita dapat mengunduh dataset yang akan kita gunakan untuk melatih model kita. Dataset ini berisi 1040 file captcha dalam bentuk gambar png. Dataset dapat Anda unduh dari tautan :")
        link = "https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip"
        columns = st.columns(1)
        for column in columns:
            column.markdown(f"[here]({link})")
        show_top_20_images()
        
    if __name__ == '__main__':
        main()
        
with tab2:
    
    def adaptive_thresholding(image, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, binary_type=cv2.THRESH_BINARY, block_size=145, constant=0):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.adaptiveThreshold(gray_image, 255, method, binary_type, block_size, constant)
        return thresh_img

    def plot_(image1, image2):
        st.image([image1, image2], caption=['Image 1', 'Image 2'], width=300)
    
    def draw_rectangles(image):
        cv2.rectangle(image, (30, 12), (50, 49), 0, 1)
        cv2.rectangle(image, (50, 12), (70, 49), 0, 1)
        cv2.rectangle(image, (70, 12), (90, 49), 0, 1)
        cv2.rectangle(image, (90, 12), (110, 49), 0, 1)
        cv2.rectangle(image, (110, 12), (130, 49), 0, 1)
        return image
    
    path = 'datasets/captcha_images_v2/samples/'
    
    def t_img (img) :
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

    def c_img (img) :
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))

    def d_img (img) :
        return cv2.dilate(img, np.ones((2,2), np.uint8), iterations = 1)

    def b_img (img) :
        return cv2.GaussianBlur(img, (1,1), 0)

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

    def main():
        
        path1 = 'datasets/captcha_images_v2/yf347.png'
        path2 = 'datasets/captcha_images_v2/yyg5g.png'

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        st.markdown(
            "<h3 style='text-align: center;'>Take a Sample</h3>", unsafe_allow_html=True)
        st.write("Mengidentifikasi setiap huruf yang muncul pada CAPTCHA yang nantinya akan dimasukkan dan dilatih menggunakan model CNN.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption='Image 1', width=300)
        with col2:
            st.image(img2, caption='Image 2', width=300)

        thresh_img1 = adaptive_thresholding(img1)
        thresh_img2 = adaptive_thresholding(img2)

        st.markdown(
            "<h3 style='text-align: center;'>Image Processing</h3>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: justify;'>1. Adaptive Thresholding</h4>", unsafe_allow_html=True)
        st.write("Algoritma ini mendapatkan ambang batas pixel yang berbeda untuk wilayah yang berbeda pada gambar yang sama, yang memberikan hasil yang lebih baik untuk gambar dengan pencahayaan yang bervariasi.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(thresh_img1, caption='Image 1', width=300)
        with col2:
            st.image(thresh_img2, caption='Image 2', width=300)

        gauss_img1 = draw_rectangles(np.copy(thresh_img1))
        gauss_img2 = draw_rectangles(np.copy(thresh_img2))
        
        st.markdown(
            "<h4 style='text-align: justify;'>2. Erosion (Pengerutan)</h4>", unsafe_allow_html=True)
        st.write("Ini akan menyusutkan area objek dan menghilangkan noise serta elemen-elemen kecil yang mungkin muncul setelah proses dilation.")
        st.markdown(
            "<h4 style='text-align: justify;'>3. Dilation (Pembesaran)</h4>", unsafe_allow_html=True)
        st.write("Melibatkan kernel yang dipindai pada seluruh gambar. Nilai piksel maksimal dihitung di wilayah kernel dan titik jangkar kernel diperbarui ke nilai tersebut. Hal ini menyebabkan wilayah putih meluas dalam gambar.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gauss_img1, caption='Image 1', width=300)
        with col2:
            st.image(gauss_img2, caption='Image 2', width=300)

        st.markdown(
            "<h4 style='text-align: justify;'>4. Smoothing Images (Blurring)</h4>", unsafe_allow_html=True)
        st.write("Melibatkan penggabungan filter low-pass dengan gambar, untuk menghilangkan komponen frekuensi tinggi. Misalnya, noise dan tepi gambar.")
        
        fig, ax = plt.subplots(figsize=(20, 5))
        for i in range(5):
            ax = plt.subplot(1, 5, i+1)
            ax.imshow(X[i], 'gray')
            ax.set_title('Label is ' + str(y[i]))
        st.pyplot(fig)
        

    if __name__ == '__main__':
        main()

with tab3:
    
    def main():
        
        st.markdown(
            "<h4 style='text-align: justify;'>Label Distribution</h4>", unsafe_allow_html=True)
        st.write("Label Distribution merujuk pada distribusi frekuensi atau proporsi dari setiap kategori atau label dalam suatu dataset atau populasi. Ini memberikan pemahaman tentang seberapa seimbang atau tidak seimbangnya distribusi kategori dalam dataset.")
        
        temp1 = set(y)
        
        temp_df1 = pd.DataFrame({'labels' : [t for t in temp1], 'Count' : [len(y[y==t]) for t in temp1]})
        
        fig, ax = plt.subplots(figsize=(20, 7))
        sns.barplot(x='labels', y='Count', data=temp_df1, palette='Blues_d', ax=ax)
        ax.set_title('Label distribution in CAPTCHA', fontsize=20)
        st.pyplot(fig)
        
    if __name__ == '__main__':
        main()
        
with tab4:
    
    def conv_layer (filterx) :
    
        model = Sequential()
    
        model.add(Conv2D(filterx, (3,3), padding = 'same', activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    
        return model
    
    def dens_layer (hiddenx) :
    
        model = Sequential()
    
        model.add(Dense(hiddenx, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    
        return model
    
    def cnn(filter1, filter2, filter3, hidden1, hidden2):
        model = Sequential()
        model.add(Input((40, 20, 1,)))
    
        model.add(Conv2D(filter1, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
        model.add(Conv2D(filter2, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
        model.add(Conv2D(filter3, (3, 3), activation='relu'))
        model.add(Flatten())
    
        model.add(Dense(hidden1, activation='relu'))
        model.add(Dense(hidden2, activation='relu'))
    
        model.add(Dense(19, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        return model
    
    st.markdown(
        "<h4 style='text-align: justify;'>Data Augmentation</h4>", unsafe_allow_html=True)
    st.write("Data Augmentation pada teks CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) mungkin berbeda dengan data augmentation pada gambar. Namun, pada dasarnya, tujuannya tetap sama: untuk meningkatkan keberagaman data yang digunakan dalam pelatihan model agar model dapat lebih baik dalam mengenali berbagai variasi teks CAPTCHA.")
        
    
    temp1 = set(y)
        
    temp_df1 = pd.DataFrame({'labels' : [t for t in temp1], 'Count' : [len(y[y==t]) for t in temp1]})
    
    y_combine = LabelEncoder().fit_transform(y)
    y_one_hot = OneHotEncoder(sparse_output = False).fit_transform(y_combine.reshape(len(y_combine),1))
        
    info = {y_combine[i] : y[i] for i in range(len(y))}
        
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.2, random_state = 1)
        
    y_temp = np.argmax(y_test, axis = 1)
        
    temp2 = set(y)
        
    temp_df2 = pd.DataFrame({'labels' : [t for t in temp2], 'Count' : [len(y[y==t]) for t in temp2]})
    
    X_train = np.reshape(X_train, (4160, 40*20*1))
    
    X_train, y_train = SMOTE(sampling_strategy = 'auto', random_state = 1).fit_resample(X_train, y_train)
    
    X_train = np.reshape(X_train, (8037, 40, 20, 1))
    
    def plot_images(X_train, y_train, info, num_images=25):
        fig = plt.figure(figsize=(20, 20))
        hi = len(X_train)
        lo = 0

        for i in range(num_images):
            plt.subplot(5, 5, i + 1)
            x = np.random.randint(lo, hi)
            plt.imshow(X_train[x], 'gray')
            plt.title('Label is ' + str(info[np.argmax(y_train[x])]))
            plt.axis('off')       
        return fig

    st.pyplot(plot_images(X_train, y_train, info))
    
    traingen = ImageDataGenerator(rotation_range = 5, width_shift_range = [-2,2])
    traingen.fit(X_train)
    
    train_set = traingen.flow(X_train, y_train)
    
    trainX, trainy = train_set.next()
    
    def plot_images(trainX, trainy, info, num_images=25):
        fig = plt.figure(figsize=(20, 20))
        hi = len(trainX)
        lo = 0

        for i in range(num_images):
            plt.subplot(5, 5, i + 1)
            x = np.random.randint(lo, hi)
            plt.imshow(trainX[x], 'gray')
            plt.title('Label is ' + str(info[np.argmax(trainy[x])]))
            plt.axis('off')
        return fig

    st.pyplot(plot_images(trainX, trainy, info))
    
with tab5:
    
    st.markdown(
        "<h4 style='text-align: center;'>Evaluation Model</h4>", unsafe_allow_html=True)
    st.write("proses mengevaluasi kinerja model machine learning atau model statistik terhadap data tes yang belum pernah dilihat oleh model selama pelatihan. Tujuannya adalah untuk mengukur seberapa baik model dapat menggeneralisasi dari data pelatihan ke data baru yang belum pernah dilihat sebelumnya.")  
    
    _left, mid, _right = st.columns([0.05, 0.9, 0.05])
    with mid:
        st.image("assets/model.png", caption="Model Image", use_column_width=True)