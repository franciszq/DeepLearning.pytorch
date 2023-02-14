import os
from . import ssd


def get_cfg(filepath):
    filename = os.path.basename(filepath)
    if filename == "ssd.py":
        model_name = filename.split('.')[0]
        return ssd.Config(), model_name
    else:
        raise ValueError(f"找不到{filepath}或{filepath}不存在")