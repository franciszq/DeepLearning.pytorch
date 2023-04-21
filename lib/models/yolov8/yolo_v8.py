import contextlib
from typing import List

import ast
import torch
import torch.nn as nn

from lib.models.yolov8.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                       Classify, Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                       Ensemble, Focus, GhostBottleneck, GhostConv, Pose, Segment)
from lib.utils.show import colorstr
from lib.utils.useful_tools import make_divisible


class Yolo8(nn.Module):
    def __init__(self, scale: List, num_classes=80, ch=3):
        """
        :param scale: [depth, width, max_channels] 不同尺寸的yolo_v8模型的参数
        :param num_classes: 类别数，默认数据集为COCO，80类
        :param ch: 输入通道数，默认为3
        """
        super().__init__()
        self.depth, self.width, self.max_channels = scale
        self.num_classes = num_classes
        layers = []

        # backbone
        layers.append(Conv(c1=ch, c2=self._ac(64), k=3, s=2))

        layers.append(Conv(c1=self._ac(64), c2=self._ac(128), k=3, s=2))
        layers.append(C2f(c1=self._ac(128), c2=self._ac(128), n=self._get_n(3), shortcut=True))

        layers.append(Conv(c1=self._ac(128), c2=self._ac(256), k=3, s=2))
        layers.append(C2f(c1=self._ac(256), c2=self._ac(256), n=self._get_n(6), shortcut=True))

        layers.append(Conv(c1=self._ac(256), c2=self._ac(512), k=3, s=2))
        layers.append(C2f(c1=self._ac(512), c2=self._ac(512), n=self._get_n(6), shortcut=True))

        layers.append(Conv(c1=self._ac(512), c2=self._ac(1024), k=3, s=2))
        layers.append(C2f(c1=self._ac(1024), c2=self._ac(1024), n=self._get_n(3), shortcut=True))

        layers.append(SPPF(c1=self._ac(1024), c2=self._ac(1024), k=5))

        # head
        layers.append(nn.Upsample(scale_factor=2))
        layers.append(Concat(dimension=1))
        layers.append(C2f(c1=(self._ac(1024) + self._ac(512)), c2=self._ac(512), n=self._get_n(3)))

        layers.append(nn.Upsample(scale_factor=2))
        layers.append(Concat(dimension=1))
        layers.append(C2f(c1=(self._ac(512) + self._ac(256)), c2=self._ac(256), n=self._get_n(3)))

        layers.append(Conv(c1=self._ac(256), c2=self._ac(256), k=3, s=2))
        layers.append(Concat(dimension=1))
        layers.append(C2f(c1=(self._ac(256) + self._ac(512)), c2=self._ac(512), n=self._get_n(3)))

        layers.append(Conv(c1=self._ac(512), c2=self._ac(512), k=3, s=2))
        layers.append(Concat(dimension=1))
        layers.append(C2f(c1=(self._ac(512) + self._ac(1024)), c2=self._ac(1024), n=self._get_n(3)))

        layers.append(Detect(nc=self.num_classes, ch=(self._ac(256), self._ac(512), self._ac(1024))))

        self.model = nn.Sequential(*layers)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = True
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

    def _get_n(self, n):
        return max(round(n * self.depth), 1) if n > 1 else n

    def _ac(self, c):
        """
        Adjust channel
        :param c: channel
        :return:
        """

        if c != self.num_classes:
            return make_divisible(min(c, self.max_channels) * self.width, 8)
        else:
            return c

    def forward(self, x):
        return x


def get_yolo8_n():
    # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    return Yolo8(scale=[0.33, 0.25, 1024])


def get_yolo8_s():
    # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    return Yolo8(scale=[0.33, 0.50, 1024])


def get_yolo8_m():
    # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    return Yolo8(scale=[0.67, 0.75, 768])


def get_yolo8_l():
    # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    return Yolo8(scale=[1.00, 1.00, 512])


def get_yolo8_x():
    # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
    return Yolo8(scale=[1.00, 1.25, 512])
