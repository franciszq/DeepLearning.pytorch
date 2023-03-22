from models.yolov7_model import Yolo7
from trainer.yolo7_train import Yolo7Trainer
from predict.yolo7_decode import yolo7_predictor


class ModelRegistry:
    def __init__(self, model_class, model_trainer, model_predictor):
        self.model_class = model_class
        self.model_trainer = model_trainer
        self.model_predictor = model_predictor


def register_model():
    MODEL_REGISTRY = {}
    # 注册yolo7模型
    MODEL_REGISTRY.update({"yolo7": ModelRegistry(Yolo7, Yolo7Trainer, yolo7_predictor)})
    return MODEL_REGISTRY
