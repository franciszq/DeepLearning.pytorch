from functools import partial
from typing import Dict, List

import torch
from tqdm import tqdm
from lib.algorithms.ssd import Ssd
from lib.data.collate import ssd_collate
from lib.data.detection_dataset import DetectionDataset
from lib.trainer.base import BaseTrainer
from torch.utils.data import DataLoader

from lib.trainer.lr_scheduler import get_optimizer, warm_up_scheduler


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
        train_dataset = DetectionDataset(dataset_name=self.dataset_name,
                                         input_shape=self.input_image_size[1:],
                                         mosaic=False,
                                         mosaic_prob=0,
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
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=partial(ssd_collate,
                               ssd_algorithm=self.model_algorithm))
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=partial(ssd_collate,
                               ssd_algorithm=self.model_algorithm))

    def set_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_name, self.model,
                                       self.initial_lr)

    def set_lr_scheduler(self):
        self.lr_scheduler = warm_up_scheduler(optimizer=self.optimizer,
                                              warmup_epochs=self.warmup_epochs,
                                              multi_step=True,
                                              milestones=self.milestones,
                                              gamma=self.gamma,
                                              last_epoch=self.last_epoch)

    def set_criterion(self):
        self.criterion = self.model_algorithm.build_loss()

    def train_loop(self, images, targets, scaler) -> List:
        images = images.to(device=self.device)
        targets = targets.to(device=self.device)

        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss, l_loss, c_loss = self.criterion(y_true=targets,
                                                      y_pred=preds)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss, l_loss, c_loss = self.criterion(y_true=targets, y_pred=preds)
            loss.backward()
            self.optimizer.step()

        return [loss, l_loss, c_loss]

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
                    loss_value, _, _ = self.criterion(y_true=targets,
                                                      y_pred=preds)

                    val_loss += loss_value.item()

        val_loss /= num_batches
        return {'val_loss': val_loss}
