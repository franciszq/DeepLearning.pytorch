import numpy as np
import torch

from lib.algorithms.ssd import Ssd

def yolo7_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i   # 图片在这个batch中的编号
        bboxes.append(box)
    images = torch.stack(images, dim=0)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes


def ssd_collate(batch, ssd_algorithm):
    images = []
    targets = []
    for i, (image, target) in enumerate(batch):
        images.append(image)
        target = ssd_algorithm.generate_targets(target)
        targets.append(target)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets
