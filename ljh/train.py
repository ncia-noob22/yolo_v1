import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
import torch.nn as nn
from utils import *
from model import YOLOv1


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, true):
        pred = pred.reshape(-1, S**2, C + 5 * B)  # N ✕ (...) -> N ✕ S^2 ✕ (C + 5B)

        ious = [
            calculate_IoU(
                pred[..., C + (1 + 5 * b) : C + (5 + 5 * b)], true[..., C + 1 : C + 5]
            )
            for b in range(B)
        ]
        ious = torch.cat([iou.unsqueeze(0) for iou in ious])

        iou_max, idx_iou_max = torch.max(ious, dim=0)
        is_obj_in = true[..., C]

        # Loss of box coordinates
        coord_pred = (
            idx_iou_max
            * pred[..., C + (1 + 5 * idx_iou_max) : C + (5 + 5 * idx_iou_max)]
        )
        coord_true = is_obj_in * true[..., C + 1 : C + 5]

        coord_pred[..., 2:4] = torch.sqrt(coord_pred[..., 2:4] + 1e-7)
        coord_true[..., 2:4] = torch.sqrt(coord_true[..., 2:4] + 1e-7)

        loss_coord = lambda_coord * self.mse(
            is_obj_in * coord_pred, is_obj_in * coord_true
        )

        # Loss of obj cells
        obj_pred = idx_iou_max * pred[..., C + 5 * idx_iou_max]
        loss_obj = self.mse(is_obj_in * obj_pred, is_obj_in * is_obj_in)

        # Loss of noobj cells
        noobj_preds = [b * pred[..., C + 5 * b] for b in range(B)]
        loss_noobj = lambda_noobj * sum(
            [
                self.mse((1 - is_obj_in) * noobj_pred, (1 - is_obj_in) * is_obj_in)
                for noobj_pred in noobj_preds
            ]
        )

        # Loss of classes
        class_pred = pred[..., :20]
        class_true = true[..., :20]

        loss_class = self.mse(is_obj_in * class_pred, is_obj_in * class_true)

        return sum([loss_coord, loss_obj, loss_noobj, loss_class])
