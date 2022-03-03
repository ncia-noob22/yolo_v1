path_pretrained = None

# model
S = 7  # num_grid
B = 2  # num_bbox
C = 20  # num_class

# loss
lambda_coord = 5
lambda_noobj = 0.5

# data
dir_data = "/data/torch/VOCdetection"
year = 2007

# train
num_epoch = 135
batch_size = 1  # 64
momentum = 0.9
weight_decay = 0.0005

# pred
ths_conf = 0.4
ths_iou = 0.5
