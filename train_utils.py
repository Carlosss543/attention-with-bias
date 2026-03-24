from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


def get_data_loaders(batch_size, img_size, device):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std for ImageNet
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std for ImageNet
    ])

    train_dataset = datasets.ImageFolder(root="data/imagenette2-320/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root="data/imagenette2-320/val", transform=val_transform)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # create a mask for filtering indices that are not in imagenette
    imagenette_indices = torch.tensor([0, 217, 482, 491, 497, 566, 569, 571, 574, 701])
    mask = torch.zeros(1000, dtype=torch.bool)
    mask[imagenette_indices] = True

    return train_loader, val_loader, mask.to(device)


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # 1. Classical loss (hard labels)
    ce_loss = F.cross_entropy(student_logits, labels)

    # 2. Soft targets with temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)

    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # 3. Combination
    loss = alpha * ce_loss + (1 - alpha) * (T * T) * kl_loss
    return loss


def compute_batch_accuracy(outputs, labels):
    predicted = torch.argmax(outputs, dim=1)
    return (predicted == labels).float().mean().item()
