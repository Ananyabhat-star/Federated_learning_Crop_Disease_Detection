import os
import numpy as np
from tensorflow.keras.models import load_model, clone_model

CLIENT_DIR = "training/client_models"
OUT_DIR = "training/models"
N_CLIENTS = 5

os.makedirs(OUT_DIR, exist_ok=True)

models = []
for i in range(N_CLIENTS):
    path = os.path.join(CLIENT_DIR, f"client_{i}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    models.append(load_model(path))

print("✔ All client models loaded")

global_model = clone_model(models[0])
global_model.set_weights(models[0].get_weights())

avg_weights = []
for weights in zip(*[m.get_weights() for m in models]):
    avg_weights.append(np.mean(weights, axis=0))

global_model.set_weights(avg_weights)

global_model.save(os.path.join(OUT_DIR, "global_model.keras"))
print("✅ Global model saved")
