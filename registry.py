from lib.algorithms.centernet import CenterNetA
from lib.algorithms.yolo_v7 import YOLOv7
from lib.algorithms.ssd import Ssd

from configs import yolo7_cfg, ssd_cfg, centernet_cfg, yolo8_det_cfg
from lib.algorithms.yolo_v8 import YOLOv8
from lib.trainer import Yolo7Trainer, SsdTrainer, CenterNetTrainer
from lib.trainer.yolo8_train import Yolo8Trainer

# 模型注册表
# key：配置文件
# value：配置文件对应的配置类，模型类，训练类
model_registry = {
    "yolo7_cfg.py": [yolo7_cfg.Config(), YOLOv7, Yolo7Trainer],
    "ssd_cfg.py": [ssd_cfg.Config(), Ssd, SsdTrainer],
    "centernet_cfg.py": [centernet_cfg.Config(), CenterNetA, CenterNetTrainer],
    "yolo8_det_cfg.py": [yolo8_det_cfg.Config(), YOLOv8, Yolo8Trainer]
}
