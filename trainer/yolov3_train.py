from typing import List

from trainer.base import BaseTrainer


class Yolo3Trainer(BaseTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

    def initialize_model(self):
        pass

    def load_data(self):
        pass

    def set_optimizer(self):
        pass

    def set_lr_scheduler(self):
        pass

    def train_loop(self, images, targets, scaler) -> List:
        pass