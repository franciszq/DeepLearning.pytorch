import numpy as np
import torch
import math


def get_random_number(a=0.0, b=1.0):
    """生成[a,b)范围内的随机数"""
    return np.random.rand() * (b - a) + a