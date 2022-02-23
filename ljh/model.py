import sys
from pathlib import Path

origin = Path(__file__).parent.parent
sys.path.append(str(origin))

import torch.nn as nn
from ljh.utils import *


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
        for layer_config in network_config:
            if layer_config[0] == "conv":
                layers.append(CNNLayer(in_channels, *layer_config[1:]))
                in_channels = layer_config[1]
            elif layer_config[0] == "maxpool":
                layers.append(nn.MaxPool2d(*layer_config[1:]))
        layers.append(FCLayer())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
