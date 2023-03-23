from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from algorithms.yolo_v7 import YOLOv7

from data.collate import yolo7_collate
from data.detection_dataset import DetectionDataset
from loss.yolo7_loss import Yolo7Loss
# from models.yolov7_model import Yolo7
from trainer.base import BaseTrainer
from utils.anchor import get_yolo7_anchors
from trainer.lr_scheduler import get_optimizer, warm_up_scheduler


class Yolo7Trainer(BaseTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.yolo_v7 = YOLOv7(cfg)
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss"]
        # 是否在tqdm进度条中显示上述metrics
        self.show_option = [True]

    def initialize_model(self):
        self.model, model_name = self.yolo_v7.build_model()
        self.model.to(device=self.device)
        self.set_model_name(model_name)

    def load_data(self):
        train_dataset = DetectionDataset(dataset_name=self.dataset_name,
                                         input_shape=self.input_image_size[1:],
                                         mosaic=True,
                                         mosaic_prob=0.5,
                                         epoch_length=self.total_epoch,
                                         special_aug_ratio=0.7,
                                         train=True)
        val_dataset = DetectionDataset(dataset_name=self.dataset_name,
                                       input_shape=self.input_image_size[1:],
                                       mosaic=False,
                                       mosaic_prob=0,
                                       epoch_length=self.total_epoch,
                                       special_aug_ratio=0,
                                       train=False)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers,
                                           drop_last=True, collate_fn=yolo7_collate)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=True, num_workers=self.num_workers,
                                         drop_last=True, collate_fn=yolo7_collate)

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model, self.initial_lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = warm_up_scheduler(optimizer=self.optimizer,
                                              warmup_epochs=self.warmup_epochs,
                                              multi_step=True,
                                              milestones=self.milestones,
                                              gamma=self.gamma,
                                              last_epoch=self.last_epoch)


    def set_criterion(self):
        self.criterion = self.yolo_v7.build_loss()


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

    def evaluate_loop(self) -> Dict:
        self.model.eval()
        val_loss = 0
        num_batches = len(self.val_dataloader)

        with tqdm(self.val_dataloader, desc="Evaluate") as pbar:
            with torch.no_grad():
                for i, (images, targets) in enumerate(pbar):
                    images = images.to(device=self.device)
                    targets = targets.to(device=self.device)
                    preds = self.model(images)
                    loss_value = self.criterion(preds, targets, images)

                    val_loss += loss_value.item()

        val_loss /= num_batches
        return {'val_loss': val_loss}