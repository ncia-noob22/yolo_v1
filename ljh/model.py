import torch.nn as nn

config_network = [
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


class ConvLayer(nn.Module):
    """Convolutional block consisting of convolution and linear activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.conv(x))


class FCLayer(nn.Module):
    """Fully connected block for prediction based on the paper"""

    def __init__(self, S, B, C):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 ** 2 * 1024, 4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S ** 2 * (C + 5 * B)),
        )

    def forward(self, x):
        return self.fc(x)


class YOLOv1(nn.Module):
    def __init__(self, S, B, C, **kwargs):
        super().__init__()
        in_channels, layers = 3, []

        # form convolutional layers
        for config_layer in config_network:
            if config_layer[0] == "conv":
                layers.append(ConvLayer(in_channels, *config_layer[1:]))
                in_channels = config_layer[1]
            elif config_layer[0] == "maxpool":
                layers.append(nn.MaxPool2d(*config_layer[1:]))

        # form fully connected layers
        layers.append(FCLayer(S, B, C))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import yaml
    import torch
    import torchsummary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    model = YOLOv1(**config).to(device)
    print(torchsummary.summary(model, (3, 448, 448), device=device.split(":")[0]))
