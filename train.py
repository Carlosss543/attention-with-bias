import torch
import torch.nn as nn
from custom_vision_transformer import vit_b_16, ViT_B_16_Weights
from custom_vision_transformer import vit_custom_16
from training_parameters import d_params
from train_utils import *
import wandb
from tqdm import tqdm


# --- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("GPU visible :", torch.cuda.current_device())
print("GPU name :", torch.cuda.get_device_name(torch.cuda.current_device()))


# --- parameters ---
batch_size = d_params["batch_size"]
img_size = d_params["img_size"]
num_classes = d_params["num_classes"]
num_epochs = d_params["num_epochs"]
learning_rate = d_params["learning_rate"]
alpha = d_params["alpha"]  # weight for distillation loss (between 0 and 1)


# --- load teacher model ---
teacher_weights = ViT_B_16_Weights.IMAGENET1K_V1
teacher_model = vit_b_16(weights=teacher_weights).to(device)
teacher_model.eval()


# --- load model ---
model = vit_custom_16(num_classes=num_classes, attention_bias=True).to(device)
model.heads = nn.Linear(in_features=model.hidden_dim, out_features=num_classes)
model.to(device)


# --- load data and get mask for filtering indices that are not in imagenette ---
train_loader, val_loader, mask = get_data_loaders(batch_size, img_size, device)
val_iter = iter(val_loader)


# --- optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# --- initialize wandb ---
# Start a new wandb run to track this script.
run = wandb.init(
    project="attention_bias",
    dir="./wandb_logs",
    config={
        "model_type": "vit_custom_16",
        "dataset": "Imagenette",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "alpha": alpha,
    },
    mode="online"  # online/disabled
)


# --- training loop ---
for epoch in range(num_epochs):
    for i, (imgs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):

        imgs, labels = imgs.to(device), labels.to(device)

        # --- teacher forward pass ---
        with torch.no_grad():
            teacher_outputs = teacher_model(imgs)
            teacher_outputs = teacher_outputs[:, mask] # filter out indices not in imagenette

        output = model(imgs)

        loss = distillation_loss(output, teacher_outputs, labels, alpha=alpha)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


        if i % 10 == 0:
            train_acc = compute_batch_accuracy(output.detach(), labels)

            try:
                imgs, labels = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                imgs, labels = next(val_iter)
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)
                teacher_outputs = teacher_outputs[:, mask]
                output = model(imgs)
                val_loss = distillation_loss(output, teacher_outputs, labels, alpha=alpha)

                val_acc = compute_batch_accuracy(output.detach(), labels)

            # send the metrics to wandb
            wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "train_acc": train_acc, "val_acc": val_acc})
