import torch


def calculate_IoU(bboxes1, bboxes2):
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


# def select_bbox_bycell(bboxes):
#     pass


# def do_NMS(bboxes, ths_conf, ths_iou):
#     bboxes_chosen = [box for box in bboxes if box[1] > ths_conf]


# def get_bboxes(dataloader, model, ths_conf, ths_iou, batch_size, device, **kwargs):
#     bboxes_pred, bboxes_true = [], []

#     model.eval()
#     for data, labels in dataloader:
#         data, labels = data.to(device), labels.to(device)

#         with torch.no_grad:
#             preds = model(data)


# def calculate_mAP(boxes_pred, boxes_true, ths_iou):
#     pass
