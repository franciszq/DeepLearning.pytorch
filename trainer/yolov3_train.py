import os
from typing import List

import torch

from data.yolov3_dataloader import Yolo3Loader
from loss.yolov3_loss import YoloLoss, make_label
from mAP.eval import evaluate_pipeline
from models.yolov3_model import YoloV3
from predict.yolov3_decode import Decoder
from trainer.base import BaseTrainer
from utils.lr_scheduler import get_optimizer, get_lr_scheduler


class Yolo3Trainer(BaseTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss", "loc_loss", "conf_loss", "prob_loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True, False, False, False]
        self.overwrite_model_name()

    def initialize_model(self):
        self.model = YoloV3(self.cfg)
        self.model.to(device=self.device)

    def load_data(self):
        self.train_dataloader = Yolo3Loader(self.cfg,
                                            self.dataset_name,
                                            self.batch_size,
                                            self.input_image_size[1:]).__call__()

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model, self.initial_lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler("multi_step", self.optimizer, self.last_epoch,
                                             milestones=self.milestones,
                                             gamma=self.gamma)

    def set_criterion(self):
        self.criterion = YoloLoss(self.cfg, self.device)

    def train_loop(self, images, targets, scaler) -> List:
        images = images.to(device=self.device)
        targets = make_label(self.cfg, targets)
        targets = [x.to(device=self.device) for x in targets]

        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss, loc_loss, conf_loss, prob_loss = self.criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss, loc_loss, conf_loss, prob_loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

        return [loss, loc_loss, conf_loss, prob_loss]

    def evaluate(self,
                 weights=None,
                 subset='val',
                 skip=False):

        # 加载权重
        if weights is not None:
            self.load_weights(weights)
        # 切换为'eval'模式
        self.model.eval()

        evaluate_pipeline(model=self.model,
                          decoder=Decoder(self.cfg,
                                          conf_threshold=0.02,
                                          device=self.device),
                          input_image_size=self.input_image_size[1:],
                          map_out_root=os.path.join(self.result_path, "map"),
                          subset=subset,
                          device=self.device,
                          skip=skip)
