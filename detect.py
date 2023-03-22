import time
import torch

from configs import get_cfg
from models import SSD, CenterNet, YoloV3, Yolo7
from predict import ssd_decode, centernet_decode, yolov3_decode, yolo7_decode
from registry import register_model
from utils.ckpt import CheckPoint

# 权重文件位置
WEIGHTS = "saves/yolov7_weights.pth"
# 测试图片路径的列表
IMAGE_PATHS = ["test/2007_002273.jpg"]
# 配置文件路径
CONFIG = "configs/yolo7_cfg.py"


def detect_images(cfg, model_class, decode_fn, device):
    """
    检测多张图片中的目标
    :param cfg: 配置信息
    :param model_class:  网络模型的类名
    :param decode_fn: 解码函数名
    :param device: 设备
    :return:
    """
    model = model_class(cfg).to(device)
    CheckPoint.load(WEIGHTS, device, model, pure=True)
    print(f"Loaded weights: {WEIGHTS}")
    for img in IMAGE_PATHS:
        decode_fn(cfg, model, img, print_on=True, save_result=True, device=device)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model_name = get_cfg(CONFIG)
    model_registry = register_model()

    try:
        class_param = model_registry[model_name].model_class
        decode_fn_param = model_registry[model_name].model_predictor
    except KeyError:
        raise ValueError(f"Unsupported model: {model_name}")
    
    detect_images(cfg, class_param, decode_fn_param, device)

    print(f"Total time: {(time.time() - t0):.2f}s")


if __name__ == '__main__':
    main()
