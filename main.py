from PIL import Image
import numpy as np
import streamlit as st
from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions

st.title("Image Classification")
img_path = 'car.jpg'

def load_model():
    return EfficientNetB0(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_image():
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        return img

# Загружаем предварительно обученную модель
model = load_model()

# Выводим форму загрузки изображения и получаем изображение
img = load_image()

# Проверяем, что изображение загружено
if img is not None:
    # Предварительная обработка изображения
    x = preprocess_image(img)

    # Распознавание изображения
    preds = model.predict(x)

    # Вывод предсказаний
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])