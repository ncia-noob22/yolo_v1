# import sys
# from pathlib import Path

# origin = Path(__file__).parent.parent
# sys.path.append(str(origin))

import torch
import torch.nn as nn
from ljh.config import *


def calculate_IOU(boxes_pred, boxes_true):
    pass


def do_NMS(bboxes, ths, ths_iou):
    pass


def calculate_mAP(boxes_pred, boxes_true, ths_iou):
    pass


class Loss(nn.Module):
    def __init__(self, lambda_coord, lambda_noobj):
        super().__init__()

    def forward(self, pred, target):
        pass
