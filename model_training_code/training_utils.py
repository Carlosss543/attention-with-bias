from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import torch
from tqdm import tqdm
import wandb
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
from model_training_code.mixup_cutmix import get_mixup_cutmix
import model_training_code.training_parameters as params


def get_attention_pruning_metrics(model):
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    pruned_ratios = []
    for module in base_model.modules():
        if hasattr(module, "last_pruned_ratio"):
            pruned_ratios.append(module.last_pruned_ratio.detach().item())

    avg_pruned_ratio = sum(pruned_ratios) / len(pruned_ratios)
    
    return avg_pruned_ratio


class ImageNetVal(Dataset):
    def __init__(self, img_dir, csv_path, synset_mapping_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.synset_to_class = {}
        with open(synset_mapping_path, 'r') as f:
            for idx, line in enumerate(f):
                synset = line.split()[0]
                self.synset_to_class[synset] = idx
        
        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            image_id = row['ImageId']
            synset = row['PredictionString'].split()[0]
            class_idx = self.synset_to_class[synset]
            img_path = os.path.join(self.img_dir, f'{image_id}.JPEG')
            self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders(batch_size, persistent_workers=True, shuffle_val=False):

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # mean and std for ImageNet

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(params.crop_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(interpolation=InterpolationMode.BILINEAR, num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(params.resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(params.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if params.dataset_name == "ImageNet100":
        train_dataset = datasets.ImageFolder("/data_fast/data_charles/imagenet100/train", transform=train_transform)
        val_dataset = datasets.ImageFolder("/data_fast/data_charles/imagenet100/val", transform=val_transform)
    elif params.dataset_name == "ImageNet1k":
        train_dataset = datasets.ImageFolder("/data_fast/data_charles/imagenet1k/ILSVRC/Data/CLS-LOC/train", transform=train_transform)
        val_dataset = ImageNetVal(img_dir="/data_fast/data_charles/imagenet1k/ILSVRC/Data/CLS-LOC/val", csv_path="/data_fast/data_charles/imagenet1k/LOC_val_solution.csv", synset_mapping_path="/data_fast/data_charles/imagenet1k/LOC_synset_mapping.txt", transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {params.dataset_name}")

    # add cutmix and mixup to the training dataset
    mixup_cutmix = get_mixup_cutmix(mixup_alpha=params.mixup_alpha, cutmix_alpha=params.cutmix_alpha, num_classes=params.num_classes, use_v2=False)
    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))
    else:
        collate_fn = default_collate

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=4, pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=2)

    # # display some image samples in results/augmented_images.png
    # sample_imgs, _ = next(iter(train_loader))
    # sample_imgs = sample_imgs[:16]  # take first 16 images
    # sample_imgs = sample_imgs.cpu().permute(0, 2, 3, 1) * torch.tensor(std).view(1, 1, 1, 3) + torch.tensor(mean).view(1, 1, 1, 3)
    # sample_imgs = torch.clamp(sample_imgs, 0, 1).numpy()
    # fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(sample_imgs[i])
    #     ax.axis('off')
    # plt.savefig("../results/augmented_images.png")
    # plt.close()

    return train_loader, val_loader


def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, log_interval=10):
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_pruning_ratio = 0.0
    total_batches = 0

    for i, (imgs, labels) in enumerate(tqdm(train_loader, total=len(train_loader), desc="Training", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            output = model(imgs)
            loss = criterion(output, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if params.clip_grad_norm is not None:
            scaler.unscale_(optimizer) # we should unscale the gradients of optimizer's assigned params if do gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        total_samples += labels.size(0)
        total_pruning_ratio += get_attention_pruning_metrics(model)
        total_batches += 1

        # send batch loss to W&B occasionally
        if (i+1) % log_interval == 0:
            wandb.log({"batch_train_loss": loss.item()})

    avg_loss = total_loss / total_samples
    avg_pruning_ratio = total_pruning_ratio / total_batches

    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"train_loss": avg_loss, "learning_rate": current_lr, "train_avg_pruned_ratio": avg_pruning_ratio})


def val_one_epoch(val_loader, model, criterion, device, wandb_log=True):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_correct_top5 = 0
    total_samples = 0
    total_pruning_ratio = 0.0
    total_batches = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, total=len(val_loader), desc="Validation", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                output = model(imgs)
                loss = criterion(output, labels)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (torch.argmax(output, dim=1) == labels).sum().item()
            total_correct_top5 += (torch.topk(output, k=5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += labels.size(0)
            total_pruning_ratio += get_attention_pruning_metrics(model)
            total_batches += 1

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_acc_top5 = total_correct_top5 / total_samples
    avg_pruning_ratio = total_pruning_ratio / total_batches

    if wandb_log:
        wandb.log({"val_loss": avg_loss, "val_acc": avg_acc, "val_acc_top5": avg_acc_top5, "val_avg_pruned_ratio": avg_pruning_ratio})

    return avg_loss, avg_acc, avg_acc_top5, avg_pruning_ratio
