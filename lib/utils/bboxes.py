import torch
import numpy as np


def xywh_to_xyxy(coords):
    """
    坐标变换
    :param coords: numpy.ndarray, 最后一维的4个数是坐标 (center_x, center_y, w, h)
    :return: numpy.ndarray, 与输入的形状一致，最后一维的格式是(xmin, ymin, xmax, ymax)
    """
    cx = coords[..., 0:1]
    cy = coords[..., 1:2]
    w = coords[..., 2:3]
    h = coords[..., 3:4]

    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2

    new_coords = np.concatenate((xmin, ymin, xmax, ymax), axis=-1)
    return new_coords


def xywh_to_xyxy_torch(coords, more=False):
    """
    坐标变换
    :param coords: torch.Tensor, 最后一维的4个数是坐标 (center_x, center_y, w, h)
    :param more: 最后一维除了前4个数是坐标外，还有更多的数
    :return: torch.Tensor, 与输入的形状一致，最后一维的格式是(xmin, ymin, xmax, ymax)
    """
    cx = coords[..., 0:1]
    cy = coords[..., 1:2]
    w = coords[..., 2:3]
    h = coords[..., 3:4]

    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2

    new_coords = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
    if more:
        new_coords = torch.cat([new_coords, coords[..., 4:]], dim=-1)
    return new_coords


def xyxy_to_xywh(coords, center=True):
    """
    坐标变换
    :param coords: numpy.ndarray, 最后一维的4个数是坐标
    :param center: True表示将(xmin, ymin, xmax, ymax)转变为(center_x, center_y, w, h)格式
                   False表示将(xmin, ymin, xmax, ymax)转变为(xmin, ymin, w, h)格式
    :return: numpy.ndarray, 与输入的形状一致
    """
    xmin = coords[..., 0:1]
    ymin = coords[..., 1:2]
    xmax = coords[..., 2:3]
    ymax = coords[..., 3:4]
    w = xmax - xmin
    h = ymax - ymin
    if center:
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        return np.concatenate((center_x, center_y, w, h), axis=-1)
    else:
        return np.concatenate((xmin, ymin, w, h), axis=-1)


def xyxy_to_xywh_torch(coords, center=True):
    """
    坐标变换
    :param coords: torch.Tensor, 最后一维的4个数是坐标
    :param center: True表示将(xmin, ymin, xmax, ymax)转变为(center_x, center_y, w, h)格式
                   False表示将(xmin, ymin, xmax, ymax)转变为(xmin, ymin, w, h)格式
    :return: torch.Tensor, 与输入的形状一致
    """
    xmin = coords[..., 0:1]
    ymin = coords[..., 1:2]
    xmax = coords[..., 2:3]
    ymax = coords[..., 3:4]
    w = xmax - xmin
    h = ymax - ymin
    if center:
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        return torch.cat((center_x, center_y, w, h), dim=-1)
    else:
        return torch.cat((xmin, ymin, w, h), dim=-1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    box_a and box_b are both expected to be int (xmin, ymin, xmax, ymax) format.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def iou_2(anchors, boxes):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors)
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    anchor_max = anchors / 2
    anchor_min = - anchor_max
    box_max = boxes / 2
    box_min = - box_max
    intersect_min = torch.maximum(anchor_min, box_min)
    intersect_max = torch.minimum(anchor_max, box_max)
    intersect_wh = intersect_max - intersect_min
    intersect_wh = torch.clamp(intersect_wh, min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_area = boxes[..., 0] * boxes[..., 1]
    union_area = anchor_area + box_area - intersect_area
    iou = intersect_area / (union_area + 1e-12)  # shape : [N, 9]
    return iou


class Iou4:
    def __init__(self, box_1, box_2):
        """

        :param box_1: Tensor, shape: (..., 4(cx, cy, w, h))
        :param box_2: Tensor, shape: (..., 4(cx, cy, w, h))
        """
        self.box_1_min, self.box_1_max = Iou4._get_box_min_and_max(box_1)
        self.box_2_min, self.box_2_max = Iou4._get_box_min_and_max(box_2)
        self.box_1_area = box_1[..., 2] * box_1[..., 3]
        self.box_2_area = box_2[..., 2] * box_2[..., 3]

    @staticmethod
    def _get_box_min_and_max(box):
        box_xy = box[..., 0:2]
        box_wh = box[..., 2:4]
        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        return box_min, box_max

    def calculate_iou(self):
        intersect_min = torch.maximum(self.box_1_min, self.box_2_min)
        intersect_max = torch.minimum(self.box_1_max, self.box_2_max)
        intersect_wh = intersect_max - intersect_min
        intersect_wh = torch.clamp(intersect_wh, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = self.box_1_area + self.box_2_area - intersect_area
        iou = intersect_area / (union_area + 1e-12)
        return iou


def truncate_array(a, n, fill_value=-1):
    """
    对多维数组a在dim=0上截断，使a的shape为[n, ...]
    :param a:
    :param n:
    :param fill_value:  填充值，默认为-1
    :return:
    """
    if len(a) > n:
        return a[:n]
    else:
        shape = a.shape
        shape = (n,) + shape[1:]
        a = np.concatenate((a, np.full(shape, fill_value, dtype=a.dtype)), axis=0)
        return a