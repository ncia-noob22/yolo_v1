import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
import torch.nn as nn
from utils import calculate_IoU


class Loss(nn.Module):
    def __init__(self, S, B, C, lambda_coord, lambda_noobj, **kwargs):
        super().__init__()
        self.sse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, true):
        pred = pred.reshape(  # N ✕ (...) -> N ✕ S^2 ✕ (C + 5B)
            -1, self.S**2, self.C + 5 * self.B
        )

        ious = [
            calculate_IoU(
                pred[..., self.C + (1 + 5 * b) : self.C + (5 + 5 * b)],
                true[..., self.C + 1 : self.C + 5],
            )
            for b in range(self.B)
        ]
        ious = torch.cat([iou.unsqueeze(0) for iou in ious])

        iou_max, idx_iou_max = torch.max(ious, dim=0)
        is_obj_in = true[..., self.C]

        # Localization Loss
        coord_pred = (
            idx_iou_max
            * pred[..., self.C + (1 + 5 * idx_iou_max) : self.C + (5 + 5 * idx_iou_max)]
        )
        coord_true = is_obj_in * true[..., self.C + 1 : self.C + 5]

        coord_pred[..., 2:4] = torch.sqrt(coord_pred[..., 2:4] + 1e-7)
        coord_true[..., 2:4] = torch.sqrt(coord_true[..., 2:4] + 1e-7)

        loss_coord = self.lambda_coord * self.sse(
            is_obj_in * coord_pred, is_obj_in * coord_true
        )

        # Confidence Loss
        obj_pred = idx_iou_max * pred[..., self.C + 5 * idx_iou_max]
        loss_obj = self.sse(is_obj_in * obj_pred, is_obj_in * is_obj_in)

        noobj_preds = [b * pred[..., self.C + 5 * b] for b in range(self.B)]
        loss_noobj = self.lambda_noobj * sum(
            [
                self.sse((1 - is_obj_in) * noobj_pred, (1 - is_obj_in) * is_obj_in)
                for noobj_pred in noobj_preds
            ]
        )

        loss_conf = loss_obj + loss_noobj

        # Classification Loss
        class_pred = pred[..., :20]
        class_true = true[..., :20]

        loss_class = self.sse(is_obj_in * class_pred, is_obj_in * class_true)

        return sum([loss_coord, loss_conf, loss_class])