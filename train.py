import torch

from trainer import Yolo3Trainer, SSDTrainer, CenterNetTrainer, Yolo7Trainer
from configs import get_cfg

# 配置文件路径
CONFIG = "configs/yolo7.py"
# 0：训练模式，1：验证模式
MODE = 0


def main():
    cfg, model_name = get_cfg(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ssd":
        m = SSDTrainer(cfg, device)
    elif model_name == "centernet":
        m = CenterNetTrainer(cfg, device)
    elif model_name == "yolov3":
        m = Yolo3Trainer(cfg, device)
    elif model_name == "yolo7":
        m = Yolo7Trainer(cfg, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if MODE == 0:
        m.train()
    elif MODE == 1:
        m.evaluate(weights="saves/ssd_voc_final.pth")
    else:
        raise ValueError(f"Unsupported mode：{MODE}")


if __name__ == '__main__':
    main()
