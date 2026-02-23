import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = "training/models/global_model.keras"
DATASET_DIR = "training/dataset/color"

IMG_SIZE = 224
BATCH_SIZE = 32

model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

y_true = generator.classes
y_pred = np.argmax(model.predict(generator), axis=1)

acc = accuracy_score(y_true, y_pred)
print("\n✅ Accuracy:", round(acc * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(generator.class_indices.keys()),
    zero_division=0
))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
