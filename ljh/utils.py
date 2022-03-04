import torch


def calculate_IoU(bboxes1, bboxes2):
    """Calculate IoU

    Args:
        bboxes1, bboxes2 (N ✕ S^2 ✕ 4 tensor): the last dimension refers to (x, y, w, h)
    """

    x_minof1 = bboxes1[..., 0] - bboxes1[..., 2] / 2
    x_maxof1 = bboxes1[..., 0] + bboxes1[..., 2] / 2
    y_minof1 = bboxes1[..., 1] - bboxes1[..., 3] / 2
    y_maxof1 = bboxes1[..., 1] + bboxes1[..., 3] / 2

    x_minof2 = bboxes2[..., 0] - bboxes2[..., 2] / 2
    x_maxof2 = bboxes2[..., 0] + bboxes2[..., 2] / 2
    y_minof2 = bboxes2[..., 1] - bboxes2[..., 3] / 2
    y_maxof2 = bboxes2[..., 1] + bboxes2[..., 3] / 2

    x_maxofmins = torch.max(x_minof1, x_minof2)
    y_maxofmins = torch.max(y_minof1, y_minof2)
    x_minofmaxes = torch.min(x_maxof1, x_maxof2)
    y_minofmaxes = torch.min(y_maxof1, y_maxof2)

    area1 = (x_maxof1 - x_minof1) * (y_maxof1 - y_minof1)
    area2 = (x_maxof2 - x_minof2) * (y_maxof2 - y_minof2)
    intersection = (x_minofmaxes - x_maxofmins) * (y_minofmaxes - y_maxofmins)
    return intersection / (area1 - area2 + intersection + 1e-7)


def select_bbox_bycell(bboxes):  #! need to add for validation script
    """Select a responsible bounding box for each cell based on IoU

    Args:
        bboxes (N ✕ S^2 ✕ (C + 5B) tensor)

    Returns:
        N ✕ S^2 ✕ (C + 5) tensor
    """
    pass


def do_NMS(bboxes, ths_conf, ths_iou):  #! need to add for validation script
    """Do non-max suppression

    Args:
        bboxes (N ✕ S^2 ✕ (C + 5B) tensor)
        ths_conf, ths_iou (int): thresholds of confidence score and IoU respectively
    """
    bboxes_after_ths1 = [box for box in bboxes if box[1] > ths_conf]
    pass


#! need to add for validation script
def calculate_mAP(bboxes_pred, bboxes_true, ths, conf, ths_iou):
    """Calculate mAP of model

    Args:
        bboxes_pred (N ✕ S^2 ✕ (C + 5) tensor)
        bboxes_true (N ✕ S^2 ✕ (C + 5) tensor)
        ths_conf, ths_iou (int): thresholds used in do_NMS function
    """
    pass
