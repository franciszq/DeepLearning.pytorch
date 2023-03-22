import os
from . import ssd_cfg, centernet_cfg, yolov3_cfg, yolo7_cfg


CFGS = {
    "ssd_cfg.py": ssd_cfg.Config(),
    "centernet_cfg.py": centernet_cfg.Config(),
    "yolov3_cfg.py": yolov3_cfg.Config(),
    "yolo7_cfg.py": yolo7_cfg.Config(),
}


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
    try: 
        return CFGS[filename], split_filename(filename)
    except KeyError:
        raise ValueError(f"Could not find {filepath}. Perhaps the corresponding module is not registered.")
