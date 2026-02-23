import os
import shutil
import random

# ==============================
# CONFIG
# ==============================

# Path to PlantVillage COLOR dataset
DATASET_DIR = os.path.join("training", "dataset", "color")

# Output folder where federated clients will be created
OUT_ROOT = os.path.join("training", "clients")

# Number of federated clients (farms)
N_CLIENTS = 5

random.seed(42)

# ==============================
# HELPERS
# ==============================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_classes(dataset_dir):
    classes = []
    for name in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, name)
        if os.path.isdir(full):
            classes.append(name)
    return sorted(classes)

def get_images(class_dir):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
    return [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.endswith(exts)
    ]

def round_robin(images, n):
    parts = [[] for _ in range(n)]
    for i, img in enumerate(images):
        parts[i % n].append(img)
    return parts

# ==============================
# MAIN LOGIC
# ==============================

def main():
    print("📂 Reading dataset from:", DATASET_DIR)

    if not os.path.exists(DATASET_DIR):
        print("❌ Dataset path not found!")
        return

    classes = list_classes(DATASET_DIR)
    print(f"✅ Found {len(classes)} classes")

    ensure_dir(OUT_ROOT)

    # Create client folders
    for i in range(N_CLIENTS):
        ensure_dir(os.path.join(OUT_ROOT, f"client_{i}"))

    # Distribute images
    for cls in classes:
        src_cls = os.path.join(DATASET_DIR, cls)
        images = get_images(src_cls)

        if not images:
            continue

        random.shuffle(images)
        splits = round_robin(images, N_CLIENTS)

        for client_id, imgs in enumerate(splits):
            dst = os.path.join(OUT_ROOT, f"client_{client_id}", cls)
            ensure_dir(dst)

            for img in imgs:
                shutil.copy2(img, dst)

    print("\n🎉 Dataset partition completed!")
    print("📁 Clients created inside:", OUT_ROOT)

if __name__ == "__main__":
    main()
