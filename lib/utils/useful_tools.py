import numpy as np
import torch
import math


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def get_random_number(a=0.0, b=1.0):
    """生成[a,b)范围内的随机数"""
    return np.random.rand() * (b - a) + a