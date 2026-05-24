import torch
import wandb
import os
from custom_vision_transformer import custom_vit_t_16, custom_vit_s_16
from model_training_code.training_utils import *
import model_training_code.training_parameters as params
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# --- distributed training setup ---
assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."

ddp = int(os.environ.get("RANK", "-1")) != -1 # os.environ["RANK"] is defined by torchrun, so if it exists then we are in DDP mode

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    master_process = (ddp_rank == 0)
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = torch.device("cuda")
    print(f"Device: {device}:{torch.cuda.current_device()} {torch.cuda.get_device_name(torch.cuda.current_device())}")


# --- load model ---
model = custom_vit_s_16(num_classes=params.num_classes, image_size=params.crop_size, use_bias=params.use_bias, bias_only=params.bias_only, bias_topk=params.bias_topk, bias_score_threshold=params.bias_score_threshold).to(device)

start_epoch = 1

if params.resume_from_checkpoint:
    assert  os.path.exists(params.checkpoint_path), f"Checkpoint path '{params.checkpoint_path}' does not exist."

    if master_process:
        print(f"Loading checkpoint '{params.checkpoint_path}'")
    
    checkpoint = torch.load(params.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
    print(f"Custom ViT number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")


# --- load data ---
batch_size_per_gpu = int(params.batch_size / ddp_world_size)
train_loader, val_loader, train_sampler, _ = get_data_loaders(batch_size_per_gpu, ddp=ddp, rank=ddp_rank, world_size=ddp_world_size)


# --- optimizer, criterion, gradient scaler for fp16 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay, fused=True)

criterion_train = torch.nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
criterion_val = torch.nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler('cuda')


# --- learning rate scheduler with warmup and cosine decay ---
warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=params.lr_warmup_decay, total_iters=params.lr_warmup_epochs)
main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_epochs - params.lr_warmup_epochs)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[params.lr_warmup_epochs])


# --- resume optimizer, lr scheduler, and scaler states if resuming from checkpoint ---
if params.resume_from_checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    del checkpoint # free up memory


# --- initialize wandb ---
config = {k: v for k, v in vars(params).items() if not k.startswith("_")}
run = wandb.init(
    project="attention_bias",
    dir="./wandb_logs",
    config=config,
    group=f"{params.dataset_name}",
    mode="online" if master_process else "disabled",  # online/disabled
    resume="allow" if params.resume_from_checkpoint else False,
)


# --- training loop ---
for epoch in tqdm(range(start_epoch, params.num_epochs + 1), desc="Epochs", disable=not master_process):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_one_epoch(train_loader, model, criterion_train, optimizer, device, scaler, ddp=ddp, master_process=master_process)
    lr_scheduler.step()
    _ = val_one_epoch(val_loader, model, criterion_val, device, ddp=ddp, master_process=master_process, wandb_log=master_process)

    if master_process and params.checkpoints_interval is not None and epoch % params.checkpoints_interval == 0:
        model_to_save = model.module if hasattr(model, "module") else model # Unwrap the model if it's wrapped by DDP
        model_to_save = model_to_save._orig_mod if hasattr(model_to_save, "_orig_mod") else model_to_save # Unwrap the model if it's wrapped by torch.compile
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

if ddp:
    destroy_process_group()
