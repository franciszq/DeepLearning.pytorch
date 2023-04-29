import numpy as np
import torch
import math


def get_random_number(a=0.0, b=1.0):
    """生成[a,b)范围内的随机数"""
    return np.random.rand() * (b - a) + a


def move_to_device(t, device):
    """
    把数据t移动到设备device上
    :param t:
    :param device:
    :return:
    """
    # t是tensor
    if isinstance(t, torch.Tensor):
        return t.to(device)
    # t是list或者tuple
    elif isinstance(t, (list, tuple)):
        return [move_to_device(v, device) for v in t]
    # t是dict
    elif isinstance(t, dict):
        return {k: move_to_device(v, device) for k, v in t.items()}
    # t是其他类型
    else:
        return t
