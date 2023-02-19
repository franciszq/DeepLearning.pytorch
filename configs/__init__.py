import os
from . import ssd, centernet, yolov3


def get_cfg(filepath):
    filename = os.path.basename(filepath)
    if filename == "ssd_model.py":
        model_name = filename.split('.')[0]
        return ssd.Config(), model_name
    elif filename == "centernet_model.py":
        model_name = filename.split('.')[0]
        return centernet.Config(), model_name
    elif filename == "yolov3_model.py":
        model_name = filename.split('.')[0]
        return yolov3.Config(), model_name
    else:
        raise ValueError(f"Could not find {filepath}. Perhaps the corresponding module is not registered.")