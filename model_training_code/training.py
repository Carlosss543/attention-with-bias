import torch
import wandb
import os
from custom_vision_transformer import custom_vit_t_16, custom_vit_s_16
from model_training_code.training_utils import *
import model_training_code.training_parameters as params


# --- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}:{torch.cuda.current_device()} {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "CPU")


# --- load model ---
model = custom_vit_t_16(num_classes=params.num_classes, image_size=params.crop_size, use_bias=params.use_bias, bias_only=params.bias_only, bias_topk=params.bias_topk, bias_score_threshold=params.bias_score_threshold).to(device)
model = torch.compile(model)
print(f"Custom ViT number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")


# --- load data ---
train_loader, val_loader = get_data_loaders(params.batch_size)


# --- optimizer, criterion, gradient scaler for fp16 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay, fused=True)

criterion_train = torch.nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
criterion_val = torch.nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler('cuda')


# --- learning rate scheduler with warmup and cosine decay ---
warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=params.lr_warmup_decay, total_iters=params.lr_warmup_epochs)
main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_epochs - params.lr_warmup_epochs)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[params.lr_warmup_epochs])


# --- initialize wandb ---
config = {k: v for k, v in vars(params).items() if not k.startswith("_")}
run = wandb.init(
    project="attention_bias",
    dir="./wandb_logs",
    config=config,
    group=f"{params.dataset_name}",
    mode="disabled"  # online/disabled
)


# --- training loop ---
for epoch in tqdm(range(1, params.num_epochs + 1), desc="Epochs"):
    train_one_epoch(train_loader, model, criterion_train, optimizer, device, scaler)
    lr_scheduler.step()
    _ = val_one_epoch(val_loader, model, criterion_val, device, wandb_log=True)

    if params.checkpoints_interval is not None and epoch % params.checkpoints_interval == 0:
        model_to_save = getattr(model, "_orig_mod", model) # Unwrap the model if it's wrapped by torch.compile
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "params": config
        }
        checkpoint_dir = f"./training_checkpoints/checkpoint{params.folder_number}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, f"{checkpoint_dir}/vit_custom_epoch_{epoch}.pth")
