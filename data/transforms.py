import numpy as np
import torch
import torchvision.transforms.functional as F

from utils.image_process import letter_box


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), torch.from_numpy(target)


class Resize:
    def __init__(self, size):
        """
        :param size: list or tuple
        """
        self.size = size

    def __call__(self, image, target):
        image, scale, paddings = letter_box(image, self.size)
        top, bottom, left, right = paddings
        target *= scale
        # xmin, xmax增加left像素
        target[:, ::2] += left
        # ymin, ymax增加top像素
        target[:, 1::2] += top
        target /= self.size[0]    # 坐标归一化到[0, 1]
        return image, target

