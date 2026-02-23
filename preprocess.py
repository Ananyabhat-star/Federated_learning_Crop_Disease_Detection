import os
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def load_client_dataset(client_path):
    """
    Loads one client's dataset (one farm).
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        client_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    dataset = dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    return dataset


def load_all_clients(clients_root):
    """
    Load datasets for all clients.
    """
    clients = {}

    for client_name in sorted(os.listdir(clients_root)):
        client_path = os.path.join(clients_root, client_name)
        if os.path.isdir(client_path):
            print(f"Loading {client_name}")
            clients[client_name] = load_client_dataset(client_path)

    return clients


if __name__ == "__main__":
    CLIENTS_DIR = os.path.join("training", "clients")
    clients_data = load_all_clients(CLIENTS_DIR)

    print("\n✔ Preprocessing complete")
    print("✔ Clients loaded:", list(clients_data.keys()))
