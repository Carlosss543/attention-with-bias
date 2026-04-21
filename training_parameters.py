# dataset_name = "ImageNet100"
# num_classes = 100

# resize_size = 256
# crop_size = 224

# num_epochs = 300
# batch_size = 128

# lr = 0.003
# lr_warmup_epochs = 30
# lr_warmup_decay = 0.033

# weight_decay = 0.3
# label_smoothing = 0.11
# mixup_alpha = 0.2
# cutmix_alpha = 1.0
# clip_grad_norm = 1.0

# model_type = "vit_custom"
# attention_bias = False

# checkpoints_interval = 50
# folder_number = 2





# dataset_name = "ImageNet100"
# num_classes = 100

# resize_size = 256
# crop_size = 224

# num_epochs = 200 #300
# batch_size = 128

# lr = 0.002 #0.003
# lr_warmup_epochs = 10 #30
# lr_warmup_decay = 0.033

# weight_decay = 0.3 #tester plus bas
# label_smoothing = 0.1
# mixup_alpha = 0.2
# cutmix_alpha = 1.0
# clip_grad_norm = 1.0

# model_type = "vit_custom"
# attention_bias = False

# checkpoints_interval = 50
# folder_number = 2





dataset_name = "TinyImageNet"
num_classes = 200

resize_size = 64
crop_size = 64

num_epochs = 100
batch_size = 256

lr = 0.0003
lr_warmup_epochs = 10
lr_warmup_decay = 0.033

weight_decay = 0.3
label_smoothing = 0.1
mixup_alpha = 0.2
cutmix_alpha = 1.0
clip_grad_norm = 1.0

model_type = "vit_custom"
attention_bias = False

checkpoints_interval = None
folder_number = 2
