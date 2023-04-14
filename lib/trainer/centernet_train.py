from lib.algorithms.centernet import CenterNetA
from lib.trainer.base import BaseTrainer


class CenterNetTrainer(BaseTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss"]

    def set_model_algorithm(self):
        self.model_algorithm = CenterNetA(self.cfg, self.device)

    def initialize_model(self):
        self.model, self.model_name = self.model_algorithm.build_model()
        self.model.to(device=self.device)

