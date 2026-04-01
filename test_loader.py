import time
import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader
import training_parameters as params
from tqdm import tqdm


def get_data_loaders(batch_size, img_size):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std for ImageNet
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = (
        wds.WebDataset("data/imagenet100_shards/train/train-{000000..000064}.tar", shardshuffle=10)
        .shuffle(1000)  # shuffle buffer
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(train_transform, lambda x: int(x))
    )

    val_dataset = (
        wds.WebDataset("data/imagenet100_shards/val/val-{000000..000002}.tar", shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(val_transform, lambda x: int(x))
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, pin_memory=True)

    return train_loader, val_loader


train_loader, val_loader = get_data_loaders(params.batch_size, params.img_size)

t0 = time.time()
for i, (imgs, labels) in tqdm(enumerate(train_loader), total=50):
    if i > 50:
        break

print("Time per batch:", (time.time()-t0)/50)
