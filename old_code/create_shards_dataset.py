import os
import random
import webdataset as wds
from tqdm import tqdm
from PIL import Image
from io import BytesIO

set_type = "val" # mettre "train" ou "val" selon le dataset à traiter
samples_per_shard = 2000
resize_shortest_side = 320  # Resize shortest side to 320px (keeps aspect ratio)

input_dir = f"data/imagenet100/{set_type}"
output_dir = f"data/imagenet100-{resize_shortest_side}_shards/{set_type}_{samples_per_shard}"
os.makedirs(output_dir, exist_ok=True)

# build list
all_samples = []
class_dirs = sorted(os.listdir(input_dir))

for class_idx, class_name in enumerate(class_dirs):
    class_path = os.path.join(input_dir, class_name)
    for img_name in os.listdir(class_path):
        all_samples.append((os.path.join(class_path, img_name), class_idx))

print("Total:", len(all_samples))

random.shuffle(all_samples)

shard_id = 0

sink = None
count = 0

for img_path, class_idx in tqdm(all_samples, desc="Creating shards"):

    if count % samples_per_shard == 0:
        if sink:
            sink.close()
        sink = wds.TarWriter(f"{output_dir}/{set_type}-{shard_id:06d}.tar")
        shard_id += 1

    key = f"{count:07d}"

    # OPTIMIZATION: Resize shortest side to reduce I/O while keeping aspect ratio
    try:
        img = Image.open(img_path)
        # Get current aspect ratio
        w, h = img.size
        if w < h:
            # Width is shortest side
            new_w = resize_shortest_side
            new_h = int(h * (resize_shortest_side / w))
        else:
            # Height is shortest side
            new_h = resize_shortest_side
            new_w = int(w * (resize_shortest_side / h))
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Re-encode to JPEG with good quality
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_data = img_bytes.getvalue()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        continue

    sink.write({
        "__key__": key,
        "jpg": img_data,
        "cls": str(class_idx),
    })

    count += 1

if sink:
    sink.close()
