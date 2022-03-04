import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import yaml
import torch
import torch.optim as optim
from model import YOLOv1
from loss import Loss
from dataset import get_dataloaders
from train import train

# from utils import get_bboxes, calculate_mAP


def main():
    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    path_pretrained = config["path_pretrained"]
    num_epoch = config["num_epoch"]

    # load data loaders
    trainloader, testloader = get_dataloaders(**config)

    # load model
    model = YOLOv1(**config).to(device)
    if path_pretrained:
        model.load_state_dict(path_pretrained)

    # load loss function
    loss = Loss(**config)

    # load optimizer and schedule learning rate
    opt = optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    criteria_lr = (int(num_epoch * 0.55), int(num_epoch * 0.77))

    def schedule_lr(epoch):
        """Change learning rate based on the paper"""
        if epoch == 0:
            return 1e-3
        elif epoch < criteria_lr[0]:
            return 1e-2
        elif epoch < criteria_lr[1]:
            return 1e-3
        else:
            return 1e-4

    sched = optim.lr_scheduler.LambdaLR(opt, schedule_lr)

    # train model
    for epoch in range(num_epoch):
        train(trainloader, model, opt, sched, loss, epoch, device)


if __name__ == "__main__":
    main()
