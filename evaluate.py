import os

import torch
from registry import model_registry
from lib.utils.ckpt import CheckPoint

# 配置文件路径
CONFIG = "configs/centernet_cfg.py"
# "voc" or "coco"
DATASET = "voc"
# 权重文件位置
WEIGHTS = "saves/CenterNet_voc_epoch-50.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = os.path.basename(CONFIG)
    try:
        model_cfg, model_class, trainer = model_registry[cfg]
    except KeyError:
        raise ValueError(f"找不到配置文件：{cfg}.")

    model_object = model_class(model_cfg, device)
    model, _ = model_object.build_model()
    model.to(device)

    # 加载模型权重
    CheckPoint.load_pure(WEIGHTS, device, model)
    print(f"Loaded weights: {WEIGHTS}")

    if DATASET == "voc":
        model_object.evaluate_on_voc(model, "result/voc", subset='val')
    elif DATASET == "coco":
        model_object.evaluate_on_coco(model, "result/coco", subset='val')
    else:
        raise ValueError(f"Unsupported dataset：{DATASET}")


if __name__ == '__main__':
    main()
