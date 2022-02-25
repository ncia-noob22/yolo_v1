import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch.nn as nn
from utils import *

config_network = [
    # ("conv", out_channels, kernel_size, stride, padding)
    #    where padding = (out_size * stride - in_size + kernel_size) // 2
    ("conv", 64, 7, 2, 3),
    ("maxpool", 2, 2),
    ("conv", 192, 3, 1, 1),
    ("maxpool", 2, 2),
    ("conv", 128, 1, 1, 0),
    ("conv", 256, 3, 1, 1),
    ("conv", 256, 1, 1, 0),
    ("conv", 512, 3, 1, 1),
    ("maxpool", 2, 2),
    *[("conv", 256, 1, 1, 0), ("conv", 512, 3, 1, 1)] * 4,
    ("conv", 512, 1, 1, 0),
    ("conv", 1024, 3, 1, 1),
    ("maxpool", 2, 2),
    *[("conv", 512, 1, 1, 0), ("conv", 1024, 3, 1, 1)] * 2,
    ("conv", 1024, 3, 1, 1),
    ("conv", 1024, 3, 2, 1),
    ("conv", 1024, 3, 1, 1),
    ("conv", 1024, 3, 1, 1),
]


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.conv(x))


class FCLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7**2 * 1024, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S**2 * (C + 5 * B)),
        )

    def forward(self, x):
        return self.fc(x)


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels, layers = 3, []
        for config_layer in config_network:
            if config_layer[0] == "conv":
                layers.append(CNNLayer(in_channels, *config_layer[1:]))
                in_channels = config_layer[1]
            elif config_layer[0] == "maxpool":
                layers.append(nn.MaxPool2d(*config_layer[1:]))
        layers.append(FCLayer())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
