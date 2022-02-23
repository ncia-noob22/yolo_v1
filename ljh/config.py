# wip config들 어떻게 입력할지
S, B, C = 7, 2, 20

network_config = [
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
