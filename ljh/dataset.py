from xml.etree import ElementTree
from PIL import Image
import torch
from torchvision.datasets import VOCDetection
import torch.utils.data as data
import torchvision.transforms as transforms


class CustomVOCDetection(VOCDetection):
    """Custom VOC Detection dataset"""

    def __getitem__(self, idx):
        classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

        img = self.transform(Image.open(self.images[idx]).convert("RGB"))
        target = self.parse_voc_xml(ElementTree.parse(self.annotations[idx]).getroot())

        img_w = float(target["annotation"]["size"]["width"])
        img_h = float(target["annotation"]["size"]["height"])

        label = torch.zeros((7**2, 25))

        for obj in target["annotation"]["object"]:
            idx_class = classes.index(obj["name"].lower())

            x_min = float(obj["bndbox"]["xmin"]) * 448 / img_w
            x_max = float(obj["bndbox"]["xmax"]) * 448 / img_w
            y_min = float(obj["bndbox"]["ymin"]) * 448 / img_h
            y_max = float(obj["bndbox"]["ymax"]) * 448 / img_h

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            cell_x = x_center // (448 / 7)
            cell_y = y_center // (448 / 7)

            x = (x_center - cell_x * (448 / 7)) / (448 / 7)
            y = (y_center - cell_y * (448 / 7)) / (448 / 7)

            w = torch.sqrt((x_max - x_min) / 448)
            h = torch.sqrt((y_max - y_min) / 448)

            idx_cell = int(cell_x + 7 * cell_y)
            label[idx_cell, idx_class] = 1
            label[idx_cell, 20] = 1
            label[idx_cell, 21] = x
            label[idx_cell, 22] = y
            label[idx_cell, 23] = w
            label[idx_cell, 24] = h

        return img, label


def get_dataloaders(dir_data, year, batch_size, only_train, **kwargs):
    transform = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    )

    if only_train:
        trainset = CustomVOCDetection(
            dir_data, year=str(year), image_set="train", transform=transform
        )
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        return trainloader, None

    else:
        trainset = CustomVOCDetection(
            dir_data, year=str(year), image_set="train", transform=transform
        )
        validset = CustomVOCDetection(
            dir_data, year=str(year), image_set="val", transform=transform
        )

        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validloader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)

        return trainloader, validloader
