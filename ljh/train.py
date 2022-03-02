import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from model import YOLOv1
from utils import *

data_dir = "~/data/torch/VOCDetection"

train_dataset07 = VOCDetection(data_dir, "2007", "train")
valid_dataset07 = VOCDetection(data_dir, "2007", "val")

train_dataset12 = VOCDetection(data_dir, "2012", "train")
valid_dataset12 = VOCDetection(data_dir, "2012", "val")
