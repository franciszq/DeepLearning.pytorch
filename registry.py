from algorithms.yolo_v7 import YOLOv7
from configs import yolo7_cfg


# 模型注册表
# key：配置文件
# value：配置文件对应的配置类，模型类
model_registry = {
    "yolo7_cfg.py": [yolo7_cfg.Config(), YOLOv7],
}
