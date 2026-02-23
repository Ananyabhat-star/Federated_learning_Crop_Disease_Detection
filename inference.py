import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "global_model.keras")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "color")
IMG_SIZE = 224

model = load_model(MODEL_PATH)

class_names = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

def predict_image(image_path):
    if not os.path.isfile(image_path):
        return {"error": "INVALID_PATH"}

    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])

    class_name = class_names[idx]
    crop, disease = class_name.split("___")

    status = "Healthy" if "healthy" in disease.lower() else "Diseased"

    return {
        "crop": crop.replace("_", " "),
        "disease": disease.replace("_", " "),
        "status": status,
        "confidence": round(confidence * 100, 2)
    }
