import os

import torch
from registry import model_registry


# 配置文件路径
CONFIG = "configs/ssd_cfg.py"
# 0：训练模式，1：验证模式
MODE = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = os.path.basename(CONFIG)
    try:
        model_cfg, model_class, trainer = model_registry[cfg]
    except KeyError:
        raise ValueError(f"找不到配置文件：{cfg}.")

    m = trainer(model_cfg, device)

    if MODE == 0:
        # 训练模式
        m.train()
    elif MODE == 1:
        # 验证模式
        m.evaluate(weights=None)
    else:
        raise ValueError(f"Unsupported mode: {MODE}")


if __name__ == '__main__':
    main()
