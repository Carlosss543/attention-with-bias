import os
import tarfile
import random
from tqdm import tqdm

set_type = "train"

input_dir = f"data/imagenet100/{set_type}"
output_dir = f"data/imagenet100_shards/{set_type}"
samples_per_shard = 2000  # taille d'un shard (~300-600MB pour HDD)

os.makedirs(output_dir, exist_ok=True)

# 1️⃣ Créer une liste globale (image_path, class_idx)
all_samples = []
for class_idx, class_name in enumerate(sorted(os.listdir(input_dir))):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        all_samples.append((img_path, class_idx))

print(f"Total images: {len(all_samples)}")

# 2️⃣ Mélanger complètement toutes les images
random.shuffle(all_samples)

# 3️⃣ Créer les shards
shard_id = 0
sample_id = 0
tar = None

with tqdm(total=len(all_samples), desc="Creating shards") as pbar:
    for img_path, class_idx in all_samples:
        # Créer un nouveau shard si nécessaire
        if sample_id % samples_per_shard == 0:
            if tar:
                tar.close()
            tar = tarfile.open(f"{output_dir}/{set_type}-{shard_id:06d}.tar", "w")
            shard_id += 1

        key = f"{sample_id:07d}"

        # Ajouter l'image
        tar.add(img_path, arcname=f"{key}.jpg")

        # Créer le label temporaire
        label_path = f"/tmp/{key}.cls"
        with open(label_path, "w") as f:
            f.write(str(class_idx))
        tar.add(label_path, arcname=f"{key}.cls")

        sample_id += 1
        pbar.update(1)

if tar:
    tar.close()
