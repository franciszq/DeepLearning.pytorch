from lib.algorithms.yolo_v7 import YOLOv7
from lib.algorithms.ssd import Ssd
from configs import yolo7_cfg, ssd_cfg
from lib.trainer import Yolo7Trainer, SsdTrainer

# 模型注册表
# key：配置文件
# value：配置文件对应的配置类，模型类，训练类
model_registry = {
    "yolo7_cfg.py": [yolo7_cfg.Config(), YOLOv7, Yolo7Trainer],
    "ssd_cfg.py": [ssd_cfg.Config(), Ssd, SsdTrainer]
}
