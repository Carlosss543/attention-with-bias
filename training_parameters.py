dataset_name = "ImageNet100"
num_classes = 100

resize_size = 256
crop_size = 224

num_epochs = 200
batch_size = 128

lr = 0.001
lr_warmup_epochs = 10
lr_warmup_decay = 0.033

weight_decay = 0.01
label_smoothing = 0.1
mixup_alpha = 0.2
cutmix_alpha = 1.0
clip_grad_norm = 1.0

model_type = "vit_custom"
use_bias = True
bias_threshold = None
bias_topk = None
bias_score_threshold = None
bias_random_prune = None
bias_only = [False]*6 + [True]*6

checkpoints_interval = 50
folder_number = 8
