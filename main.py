from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import numpy as np
from keras.applications import EfficientNetB0
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, decode_predictions

app = FastAPI()

def load_model():
    return EfficientNetB0(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_image(file):
    image = Image.open(io.BytesIO(file))
    return image

@app.post("/predict/")
async def predict(file: UploadFile):
    model = load_model()
    img = load_image(await file.read())

    # Предварительная обработка изображения
    x = preprocess_image(img)

    # Распознавание изображения
    preds = model.predict(x)

    # Вывод предсказаний
    classes = decode_predictions(preds, top=3)[0]

    return {
        "predictions": [{"class": cl[1], "confidence": float(cl[2])} for cl in classes]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
