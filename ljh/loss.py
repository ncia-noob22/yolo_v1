import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
import torch.nn as nn
from utils import calculate_IoU


class Loss(nn.Module):
    """Loss function based on the paper

    Loss consists of localization loss, confidence losses, and classification loss
    """

    def __init__(self, S, B, C, lambda_coord, lambda_noobj, **kwargs):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.sse = nn.MSELoss(reduction="sum")

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

        _, idx_responsible = torch.max(ious, dim=0)
        idx_responsible = idx_responsible.unsqueeze(2)
        is_obj_in = true[..., self.C].unsqueeze(2)

        # Localization Loss
        coord_pred = (  #! need to adjust for B > 2
            idx_responsible * pred[..., 26:30]
            + (1 - idx_responsible) * pred[..., 21:25]
        )
        coord_true = is_obj_in * true[..., self.C + 1 : self.C + 5]

        loss_coord = self.lambda_coord * self.sse(
            is_obj_in * coord_pred, is_obj_in * coord_true
        )

        # Confidence Loss
        obj_pred = idx_responsible * pred[  #! need to adjust for B > 2
            ..., 25
        ].unsqueeze(2) + (1 - idx_responsible) * pred[..., 20].unsqueeze(2)
        loss_obj = self.sse(is_obj_in * obj_pred, is_obj_in * is_obj_in)

        noobj_preds = [
            b * pred[..., self.C + 5 * b].unsqueeze(2) for b in range(self.B)
        ]
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
