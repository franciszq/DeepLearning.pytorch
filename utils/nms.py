import torch
from torchvision.ops import nms

from utils.iou import box_diou


def diou_nms(boxes, scores, iou_threshold):
    """

    :param boxes: (Tensor[N, 4]) – boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
    :param scores: (Tensor[N]) – scores for each one of the boxes
    :param iou_threshold: (float) – discards all overlapping boxes with DIoU > iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by DIoU-NMS, sorted in decreasing order of scores
    """
    order = torch.argsort(scores, dim=0, descending=True)
    keep = list()
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            index = order[0]
            keep.append(index)
        value = box_diou(boxes1=boxes[index], boxes2=boxes[order[1:]])
        mask_index = (value <= iou_threshold).nonzero().squeeze()
        if mask_index.numel() == 0:
            break
        order = order[mask_index + 1]
    return torch.LongTensor(keep)


def gather_op(tensor, indice, device):
    """

    :param tensor: shape: (M, N)
    :param indice: shape: (K,)
    :return: Tensor, shape: (K, N)
    """
    assert tensor.dim() == 1 or tensor.dim() == 2
    if tensor.dim() == 2:
        M, N = tensor.size()
    if tensor.dim() == 1:
        M = tensor.size()[0]
        N = 1
    K = indice.size()[0]
    container = torch.zeros(K, N, dtype=torch.float32, device=device)
    for k in range(K):
        container[k] = tensor[indice[k]]
    return container


def yolo3_nms(num_classes,
              conf_threshold,
              iou_threshold,
              boxes,
              scores,
              device):
    mask = scores >= conf_threshold

    box_list = list()
    score_list = list()
    class_list = list()

    for i in range(num_classes):
        box_of_class = boxes[mask[:, i]]
        score_of_class = scores[mask[:, i], i]
        indices = nms(boxes=box_of_class, scores=score_of_class, iou_threshold=iou_threshold)
        selected_boxes = gather_op(box_of_class, indices, device)
        selected_scores = gather_op(score_of_class, indices, device)
        select_classes = torch.ones(*selected_scores.size(), dtype=torch.int32, device=device) * i

        box_list.append(selected_boxes)
        score_list.append(selected_scores)
        class_list.append(select_classes)

    boxes = torch.cat(box_list, dim=0)
    scores = torch.cat(score_list, dim=0)
    classes = torch.cat(class_list, dim=0)

    classes = torch.squeeze(classes, dim=1)

    return boxes, scores, classes
