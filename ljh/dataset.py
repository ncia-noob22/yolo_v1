from torchvision.datasets import VOCDetection
import torch.utils.data as data
import torchvision.transforms as transforms


def get_dataloaders(dir_data, year, batch_size, **kwargs):
    transform = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    )

    trainset = VOCDetection(
        dir_data, year=str(year), image_set="train", transform=transform
    )
    validset = VOCDetection(
        dir_data, year=str(year), image_set="val", transform=transform
    )

    trainloader = data.DataLoader(trainset, batch_size=batch_size, shffle=True)
    validloader = data.DataLoader(validset, batch_size=batch_size, shffle=True)

    return trainloader, validloader
