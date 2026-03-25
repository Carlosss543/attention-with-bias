from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


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

    # create a mask for filtering indices that are not present
    present_indices = torch.tensor([0, 217, 482, 491, 497, 566, 569, 571, 574, 701]) # for Imagenette
    # present_indices = torch.tensor([193, 182, 258, 162, 155, 167, 159, 273, 207, 229]).sort().values # for Imagewoof
    mask = torch.zeros(1000, dtype=torch.bool)
    mask[present_indices] = True

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


def train_one_epoch(train_loader, model, teacher_model, optimizer, device, alpha, mask, num_classes):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for imgs, labels in tqdm(train_loader, total=len(train_loader), desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # --- teacher forward pass ---
        with torch.no_grad():
            if alpha < 1.0: # only compute teacher outputs if we need them for the distillation loss
                teacher_outputs = teacher_model(imgs)
                teacher_outputs = teacher_outputs[:, mask] # filter out indices not in imagenette
            else:
                teacher_outputs = torch.zeros((imgs.size(0), num_classes), device=device) # dummy tensor since we won't use the teacher outputs when alpha=1.0

        output = model(imgs)
        loss = distillation_loss(output, teacher_outputs, labels, alpha=alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (output.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0) 

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    wandb.log({"train_loss": avg_loss, "train_acc": avg_acc})


def val_one_epoch(val_loader, model, teacher_model, device, alpha, mask, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, total=len(val_loader), desc="Validation", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            if alpha < 1.0: # only compute teacher outputs if we need them for the distillation loss
                teacher_outputs = teacher_model(imgs)
                teacher_outputs = teacher_outputs[:, mask] # filter out indices not in imagenette
            else:
                teacher_outputs = torch.zeros((imgs.size(0), num_classes), device=device) # dummy tensor since we won't use the teacher outputs when alpha=1.0

            output = model(imgs)
            loss = distillation_loss(output, teacher_outputs, labels, alpha=alpha)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (torch.argmax(output, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    wandb.log({"val_loss": avg_loss, "val_acc": avg_acc})
