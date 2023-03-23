import torch
from registry import register_model

from configs import get_cfg

# 配置文件路径
CONFIG = "configs/yolo7_cfg.py"
# 0：训练模式，1：验证模式
MODE = 0


def main():
    cfg, model_name = get_cfg(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_registry = register_model()

    try:
        trainer = model_registry[model_name].model_trainer
    except KeyError:
        raise ValueError(f"Unsupported model: {model_name}")

    m = trainer(cfg, device)

    if MODE == 0:
        # 训练模式
        m.train()
    elif MODE == 1:
        # 验证模式
        m.evaluate(weights=None)
    else:
        raise ValueError(f"Unsupported mode：{MODE}")


if __name__ == '__main__':
    main()
