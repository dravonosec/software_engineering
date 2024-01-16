import pytest
from PIL import Image
import numpy as np
from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions

@pytest.fixture
def test_image():
    # Создаем тестовое изображение размером 224x224 пикселя
    img = Image.new('RGB', (224, 224))
    return img

def test_preprocess_image(test_image):
    # Проверяем, что функция preprocess_image правильно преобразует тестовое изображение
    img = test_image
    x = preprocess_image(img)
    assert x.shape == (1, 224, 224, 3)
    assert np.min(x) >= -1 and np.max(x) <= 1

def test_image_recognition(test_image):
    # Проверяем, что модель успешно распознает тестовое изображение
    img = test_image
    x = preprocess_image(img)
    model = load_model()
    predictions = model.predict(x)
    top_classes = np.argsort(predictions)[0, -3:][::-1]
    assert len(top_classes) == 3

def load_model():
    return EfficientNetB0(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Определение теста для проверки загрузки модели
def test_load_model():
    model = load_model()
    assert model is not None

# Определение теста для предобработки изображения
def test_preprocess_image():
    img = Image.new('RGB', (224, 224))
    processed_img = preprocess_image(img)
    assert processed_img.shape == (1, 224, 224, 3)

# Определение теста для проверки распознавания изображения
def test_image_recognition():
    img = Image.open("car.jpg")
    model = load_model()
    x = preprocess_image(img)
    preds = model.predict(x)
    classes = decode_predictions(preds, top=3)[0]
    assert len(classes) == 3