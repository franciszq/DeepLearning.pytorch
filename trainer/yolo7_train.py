from typing import List

import torch
from torch.utils.data import DataLoader

from data.collate import yolo7_collate
from data.detection_dataset import DetectionDataset
from loss.yolo7_loss import Yolo7Loss
from models.yolov7_model import Yolo7
from trainer.base import BaseTrainer
from utils.anchor import get_yolo7_anchors
from trainer.lr_scheduler import get_optimizer, get_lr_scheduler


class Yolo7Trainer(BaseTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True]
        self.overwrite_model_name()

    def initialize_model(self):
        self.model = Yolo7(self.cfg)
        self.model.to(device=self.device)

    def load_data(self):
        train_dataset = DetectionDataset(dataset_name=self.dataset_name,
                                         input_shape=self.input_image_size[1:],
                                         mosaic=True,
                                         mosaic_prob=0.5)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers,
                                           drop_last=True, collate_fn=yolo7_collate)

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model, self.initial_lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(self.cfg.optimizer.scheduler_name,
                                             self.optimizer, self.last_epoch,
                                             milestones=self.milestones,
                                             gamma=self.gamma)


    def set_criterion(self):
        self.criterion = Yolo7Loss(anchors=get_yolo7_anchors(self.cfg),
                                   num_classes=self.num_classes,
                                   input_shape=self.input_image_size[1:],
                                   anchors_mask=self.cfg.arch.anchors_mask,
                                   label_smoothing=self.cfg.loss.label_smoothing)


    def train_loop(self, images, targets, scaler) -> List:
        images = images.to(device=self.device)
        targets = targets.to(device=self.device)

        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss = self.criterion(preds, targets, images)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss = self.criterion(preds, targets, images)
            loss.backward()
            self.optimizer.step()

        return [loss]