from lib.algorithms.ssd import Ssd
from lib.trainer.base import BaseTrainer


class SsdTrainer(BaseTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss", "loc_loss", "conf_loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True, True, True]

    def set_model_algorithm(self):
        self.model_algorithm = Ssd(self.cfg, self.device)

    def initialize_model(self):
        self.model, self.model_name = self.model_algorithm.build_model()
        self.model.to(device=self.device)


    def load_data(self):
        pass

    def set_optimizer(self):
        pass

    def set_lr_scheduler(self):
        pass

    def set_criterion(self):
        pass
