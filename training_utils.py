import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import training_parameters as params


def get_data_loaders(batch_size, img_size, device):

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, pin_memory=True)

    # create a mask for filtering indices that are not present
    # present_indices = torch.tensor([0, 217, 482, 491, 497, 566, 569, 571, 574, 701]) # for Imagenette
    # present_indices = torch.tensor([193, 182, 258, 162, 155, 167, 159, 273, 207, 229]).sort().values # for Imagewoof
    present_indices = torch.tensor([
    117,  70,  88, 133,   5,  97,  42,  60,  14,   3,
    130,  55,  26,   0,  89, 127,  36,  67, 110,  65,
    123,  57,  22,  21,   1,  71,  99,  16,  19, 108,
     18,  35, 124,  90,  74, 129, 125,   2,  64,  92,
    138,  48,  54,  39,  56,  96,  84,  73,  77,  52,
     20, 118, 111,  59, 106,  75, 143,  80, 140,  11,
    113,   4,  28,  50,  38, 104,  24, 107, 100,  81,
     94,  41,  68,   8,  66, 146,  29,  32, 137,  33,
    141, 134,  78, 150,  76,  61, 112,  83, 144,  91,
    135, 116,  72,  34,   6, 119,  46, 115,  93,   7
    ]).sort().values # for Imagenet100
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


def train_one_epoch(train_loader, model, teacher_model, optimizer, device, alpha, mask, scaler, scheduler, log_interval=10):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, total=params.num_train_samples//params.batch_size, desc="Training", leave=False)):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            output = model(imgs)
            if alpha < 1.0:
                with torch.no_grad():
                    teacher_outputs = teacher_model(imgs)[:, mask]
                loss = distillation_loss(output, teacher_outputs, labels, alpha=alpha)
            else:
                loss = F.cross_entropy(output, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (output.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        # send batch loss to W&B occasionally
        if (batch_idx + 1) % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"batch_train_loss": loss.item(), "learning_rate": current_lr})

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    wandb.log({"train_loss": avg_loss, "train_acc": avg_acc})


def val_one_epoch(val_loader, model, teacher_model, device, alpha, mask):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, total=params.num_val_samples//params.batch_size, desc="Validation", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                output = model(imgs)
                if alpha < 1.0:
                    teacher_outputs = teacher_model(imgs)[:, mask]
                    loss = distillation_loss(output, teacher_outputs, labels, alpha=alpha)
                else:
                    loss = F.cross_entropy(output, labels)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (torch.argmax(output, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    wandb.log({"val_loss": avg_loss, "val_acc": avg_acc})
