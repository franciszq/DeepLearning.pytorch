import os
from . import ssd, centernet


def get_cfg(filepath):
    filename = os.path.basename(filepath)
    if filename == "ssd.py":
        model_name = filename.split('.')[0]
        return ssd.Config(), model_name
    elif filename == "centernet.py":
        model_name = filename.split('.')[0]
        return centernet.Config(), model_name
    else:
        raise ValueError(f"Could not find {filepath}. Perhaps the corresponding module is not registered.")