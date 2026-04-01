import torch
from custom_vision_transformer import vit_b_16, ViT_B_16_Weights
from custom_vision_transformer import vit_custom_16
from training_utils import *
import training_parameters as params


# --- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)
print("GPU visible :", torch.cuda.current_device())
print("GPU name :", torch.cuda.get_device_name(torch.cuda.current_device()))


# --- load teacher model ---
if params.alpha < 1.0: # only load teacher model if we need it for the distillation loss
    teacher_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    teacher_model.eval()
else:
    teacher_model = None # dummy variable


# --- load model ---
attention_bias = False
model = vit_custom_16(num_classes=params.num_classes, attention_bias=attention_bias).to(device)
save_checkpoints = False
print(f"Custom ViT number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")


# --- load data and get mask for filtering indices that are not in imagenette ---
train_loader, val_loader, mask = get_data_loaders(params.batch_size, params.img_size, device)
val_iter = iter(val_loader)


# --- optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)


# --- initialize wandb ---
run = wandb.init(
    project="attention_bias",
    dir="./wandb_logs",
    config={
        "model_type": "vit_custom_16",
        "dataset": params.dataset_name,
        "batch_size": params.batch_size,
        "learning_rate": params.learning_rate,
        "alpha": str(params.alpha),
        "attention_bias": str(attention_bias),
    },
    mode="disabled"  # online/disabled
)


# --- training loop ---
for epoch in tqdm(range(1, params.num_epochs + 1), desc="Epochs"):
    train_one_epoch(train_loader, model, teacher_model, optimizer, device, params.alpha, mask, params.num_classes)
    val_one_epoch(val_loader, model, teacher_model, device, params.alpha, mask, params.num_classes)

    if epoch % 10 == 0 and save_checkpoints: # Save the model checkpoint every 10 epochs
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"./training_checkpoints/vit_custom_16_epoch_{epoch}.pth")
        print(f"Checkpoint epoch {epoch} sauvegardé.")
