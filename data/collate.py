import numpy as np
import torch


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