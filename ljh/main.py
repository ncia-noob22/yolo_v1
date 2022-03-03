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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    path_pretrained = config["path_pretrained"]
    num_epoch = config["num_epoch"]

    trainloader, testloader = get_dataloaders(**config)

    model = YOLOv1(**config).to(device)
    if path_pretrained:
        model.load_state_dict(path_pretrained)

    loss = Loss(**config)

    criteria = (
        int(num_epoch * 0.55),
        int(num_epoch * 0.77),
    )
    # max_mean_avg_precision = 0
    for epoch in range(num_epoch):
        if epoch == 0:
            lr = 1e-3
        elif epoch < criteria[0]:
            lr = 1e-2
        elif epoch < criteria[1]:
            lr = 1e-3
        else:
            lr = 1e-4

        opt = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

        # boxes_pred, boxes_true = get_bboxes(trainloader, model, **config, device=device)

        # mean_avg_precision = calculate_mAP(boxes_pred, boxes_true, **config)
        # print(f"mAP is {mean_avg_precision} for {epoch}th epoch")

        train(trainloader, model, opt, loss, device)

        # if mean_avg_precision > max_mean_avg_precision:
        #     max_mean_avg_precision = mean_avg_precision
        #     torch.save(model.state_dict(), "output/max_mAP.pt")


if __name__ == "__main__":
    main()
