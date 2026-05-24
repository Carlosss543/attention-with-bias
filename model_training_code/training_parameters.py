dataset_name = "ImageNet100"
num_classes = 100

resize_size = 256
crop_size = 224

num_epochs = 300
batch_size = 256 # 256 est le max pour vit_s_16 sur williams ou pour vit_t_16 sur frost
accumulation_steps = 4

lr = 0.001
lr_warmup_epochs = 5
lr_warmup_decay = 0.033

weight_decay = 0.05
label_smoothing = 0.1
mixup_alpha = 0.2
cutmix_alpha = 1.0
clip_grad_norm = 1.0

model_type = "custom_vit_s_16"
use_bias = False
bias_only = None
bias_topk = None
bias_score_threshold = None

checkpoints_interval = 10
folder_number = 11

resume_from_checkpoint = False
checkpoint_path = "training_checkpoints/checkpoint11/vit_custom_epoch_3.pth"
