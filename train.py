import os

import torch
from registry import model_registry


# 配置文件路径
CONFIG = "configs/yolo8_det_cfg.py"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = os.path.basename(CONFIG)
    try:
        model_cfg, model_class, trainer = model_registry[cfg]
    except KeyError:
        raise ValueError(f"找不到配置文件：{cfg}.")

    trainer(model_cfg, device).train()


if __name__ == '__main__':
    main()
