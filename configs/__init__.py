import os
from . import ssd_cfg, centernet_cfg, yolov3_cfg, yolo7_cfg


def split_filename(filename):
    """
    从文件名中分离出模型名
    :param filename: 文件名
    :return: 模型名
    """
    prefix = filename.split('.')[0]
    model_name = prefix.split('_')[0]
    return model_name


def get_cfg(filepath):
    filename = os.path.basename(filepath)
    if filename == "ssd_cfg.py":
        return ssd_cfg.Config(), split_filename(filename)
    elif filename == "centernet_cfg.py":
        return centernet_cfg.Config(), split_filename(filename)
    elif filename == "yolov3_cfg.py":
        return yolov3_cfg.Config(), split_filename(filename)
    elif filename == "yolo7_cfg.py":
        return yolo7_cfg.Config(), split_filename(filename)
    else:
        raise ValueError(f"Could not find {filepath}. Perhaps the corresponding module is not registered.")