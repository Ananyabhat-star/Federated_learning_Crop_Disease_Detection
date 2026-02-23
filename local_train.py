import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

CLIENTS_DIR = "training/clients"
OUT_DIR = "training/client_models"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

os.makedirs(OUT_DIR, exist_ok=True)

def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

clients = sorted(os.listdir(CLIENTS_DIR))

for client in clients:
    client_path = os.path.join(CLIENTS_DIR, client)
    if not os.path.isdir(client_path):
        continue

    print(f"\n🚜 Training {client}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        client_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        client_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    model = build_model(train_gen.num_classes)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(os.path.join(OUT_DIR, f"{client}.keras"))
    print(f"✔ Saved {client}.keras")
