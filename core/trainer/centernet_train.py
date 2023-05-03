from functools import partial
import torch
from typing import List, Dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from core.algorithms.centernet import CenterNetA
from core.data.collate import centernet_collate
from core.data.detection_dataset import DetectionDataset
from core.trainer.base import DetectionTrainer, use_pretrained_model
from core.trainer.lr_scheduler import get_optimizer, warm_up_scheduler
from core.utils.useful_tools import move_to_device


class CenterNetTrainer(DetectionTrainer):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        # 损失函数的返回值要与这里的metrics_name一一对应
        self.metric_names = ["loss"]
        self.show_option = [True]

    def set_model_algorithm(self):
        self.model_algorithm = CenterNetA(self.cfg, self.device)

    @use_pretrained_model
    def initialize_model(self):
        self.model, self.model_name = self.model_algorithm.build_model()
        self.model.to(device=self.device)

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
                                           drop_last=True, collate_fn=partial(centernet_collate,
                                                                              centernet_algorithm=self.model_algorithm))
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=True, num_workers=self.num_workers,
                                         drop_last=True, collate_fn=partial(centernet_collate,
                                                                            centernet_algorithm=self.model_algorithm))

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
        self.criterion = self.model_algorithm.build_loss()

    def train_loop(self, batch_data, scaler) -> List:
        images = move_to_device(batch_data[0], self.device)
        targets = move_to_device(batch_data[1], self.device)

        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                preds = self.model(images)
                loss = self.criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            preds = self.model(images)
            loss = self.criterion(preds, targets)
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
                    targets = [target.to(device=self.device) for target in targets]
                    preds = self.model(images)
                    loss_value = self.criterion(preds, targets)

                    val_loss += loss_value.item()

        val_loss /= num_batches
        return {'val_loss': val_loss}
