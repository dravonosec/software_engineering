import io
from PIL import Image
import numpy as np

from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions


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
    uploaded_file = image.load_img(img_path, target_size=(224, 224))
    return uploaded_file



# Загружаем предварительно обученную модель
model = load_model()
# Выводим форму загрузки изображения и получаем изображение
img = load_image()
# Предварительная обработка изображения
x = preprocess_image(img)
# Распознавание изображения
preds = model.predict(x)

# Вывод предсказаний
classes = decode_predictions(preds, top=3)[0]
for cl in classes:
    print(cl[1], cl[2])

# sports_car 0.6669253
# convertible 0.13077284
# racer 0.07213001