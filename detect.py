import os
import time
import torch

# from configs import get_cfg
from registry import model_registry
from utils.ckpt import CheckPoint

# 权重文件位置
WEIGHTS = "saves/yolov7_weights.pth"
# 测试图片路径的列表
IMAGE_PATHS = ["test/2007_002273.jpg"]
# 配置文件路径
CONFIG = "configs/yolo7_cfg.py"


def detect_images(model, decode_fn, device):
    """
    检测多张图片中的目标
    :param model_class:  网络模型的类名
    :param decode_fn: 解码函数名
    :param device: 设备
    :return:
    """
    CheckPoint.load(WEIGHTS, device, model, pure=True)
    print(f"Loaded weights: {WEIGHTS}")
    for img in IMAGE_PATHS:
        decode_fn(model, img, print_on=True, save_result=True)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = os.path.basename(CONFIG)
    try:
        model_cfg, model_class = model_registry[cfg]
    except KeyError:
        raise ValueError(f"找不到配置文件：{cfg}.")

    model_object = model_class(model_cfg, device)
    model, _ = model_object.build_model()
    model.to(device)

    detect_images(model, model_object.predict, device)

    print(f"Total time: {(time.time() - t0):.2f}s")


if __name__ == '__main__':
    main()
