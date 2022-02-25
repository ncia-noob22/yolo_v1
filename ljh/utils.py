import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
from config import *


def calculate_IoU(boxes_pred, boxes_true):
    pass


def do_NMS(bboxes, ths, ths_iou):
    pass


def calculate_mAP(boxes_pred, boxes_true, ths_iou):
    pass
