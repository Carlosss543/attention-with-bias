import os
import torch
from types import SimpleNamespace
import csv
from statistics import mean
from custom_vision_transformer import custom_vit_t_16, custom_vit_s_16
from model_analysis_code.utils import convert_qkv_to_q_k_v
from model_training_code.training_utils import val_one_epoch, get_data_loaders

params = SimpleNamespace(
    num_classes=100,
    img_size=224,
    batch_size=128,
    checkpoint_path="training_checkpoints/checkpoint12/vit_custom_epoch_300.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}:{torch.cuda.current_device()} {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "CPU")

_, val_loader, _, _ = get_data_loaders(batch_size=params.batch_size, persistent_workers=True, shuffle_val=True)

def evaluate_pruning(file_path, values, **bias_kwargs):
    os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure the directory exists

    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["threshold", "threshold_mean", "accuracy", "top5", "pruned_ratio"])

        active_bias = {k: v for k, v in bias_kwargs.items() if v is not None and v is not False} # Find which parameter is active
        assert len(active_bias) == 1, "Exactly one bias configuration must be provided."
        bias_key = list(active_bias.keys())[0]
        
        for threshold in values:
            kwargs = dict(num_classes=params.num_classes, image_size=params.img_size, use_bias=True, bias_only=None, bias_topk=None, bias_score_threshold=None)
            kwargs[bias_key] = threshold

            model = custom_vit_t_16(**kwargs).to(device)
            checkpoint = torch.load(params.checkpoint_path, map_location=device)
            checkpoint = checkpoint["model_state_dict"]
            checkpoint = convert_qkv_to_q_k_v(checkpoint)
            model.load_state_dict(checkpoint)
            model.eval()

            _, avg_acc, avg_acc_top5, avg_pruned_ratio = val_one_epoch(val_loader, model, torch.nn.CrossEntropyLoss(), device, wandb_log=False)

            threshold_mean = mean(threshold)

            writer.writerow([[f"{e:.3f}" for e in threshold], f"{threshold_mean:.3f}", f"{avg_acc:.4f}", f"{avg_acc_top5:.4f}", f"{avg_pruned_ratio:.4f}"])
            f.flush()  # Ensure data is written to file immediately

            print(f"Threshold: {threshold_mean:.3f}, Top-1 Accuracy: {avg_acc:.4f}, Top-5 Accuracy: {avg_acc_top5:.4f}, Pruned Ratio: {avg_pruned_ratio:.4f}")


# Generating values for pruning

# pruning_thresholds = [[1.0]*12, [0.95]*12, [0.9]*12, [0.85]*12, [0.8]*12, [0.75]*12, [0.7]*12,
#                       [1.0]*6 + [0.95]*6, [1.0]*6 + [0.9]*6, [1.0]*6 + [0.8]*6, [1.0]*6 + [0.7]*6, [1.0]*6 + [0.6]*6, [1.0]*6 + [0.5]*6, [1.0]*6 + [0.4]*6, [1.0]*6 + [0.3]*6, [1.0]*6 + [0.2]*6,
#                       torch.linspace(1.0, 0.95, steps=12).tolist(), torch.linspace(1.0, 0.9, steps=12).tolist(), torch.linspace(1.0, 0.8, steps=12).tolist(), torch.linspace(1.0, 0.7, steps=12).tolist(), torch.linspace(1.0, 0.6, steps=12).tolist(), torch.linspace(1.0, 0.5, steps=12).tolist(), torch.linspace(1.0, 0.4, steps=12).tolist(), torch.linspace(1.0, 0.3, steps=12).tolist()]

pruning_thresholds = [[1.0]*12, [0.95]*12, [0.9]*12, [0.85]*12, [0.8]*12, [0.75]*12, [0.7]*12, [0.65]*12, [0.6]*12, [0.55]*12, [0.5]*12, [0.45]*12, [0.4]*12,
                      [1.0]*6 + [0.9]*6, [1.0]*6 + [0.8]*6, [1.0]*6 + [0.7]*6, [1.0]*6 + [0.6]*6, [1.0]*6 + [0.5]*6, [1.0]*6 + [0.4]*6, [1.0]*6 + [0.3]*6, [1.0]*6 + [0.2]*6, [1.0]*6 + [0.1]*6,
                      torch.linspace(1.0, 0.9, steps=12).tolist(), torch.linspace(1.0, 0.8, steps=12).tolist(), torch.linspace(1.0, 0.7, steps=12).tolist(), torch.linspace(1.0, 0.6, steps=12).tolist(), torch.linspace(1.0, 0.5, steps=12).tolist(), torch.linspace(1.0, 0.4, steps=12).tolist(), torch.linspace(1.0, 0.3, steps=12).tolist(), torch.linspace(1.0, 0.2, steps=12).tolist(), torch.linspace(1.0, 0.1, steps=12).tolist()]

# # generate random thresholds for more variability
# values_range = torch.arange(0.2, 1.05, 0.05)
# for i in range(10):
#     random_indices = torch.randint(0, len(values_range), (12,))
#     samples = values_range[random_indices]
#     pruning_thresholds.append(samples.tolist())

method = "bias_topk"
evaluate_pruning(f"./results/pruning_graphs/{method}.csv", pruning_thresholds, bias_topk=True)
