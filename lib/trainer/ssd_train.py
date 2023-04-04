from functools import partial
from lib.algorithms.ssd import Ssd
from lib.data.collate import ssd_collate
from lib.data.detection_dataset import DetectionDataset
from lib.trainer.base import BaseTrainer
from torch.utils.data import DataLoader


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
                                           drop_last=True, collate_fn=partial(ssd_collate, ssd_algorithm=self.model_algorithm))
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=True, num_workers=self.num_workers,
                                         drop_last=True, collate_fn=partial(ssd_collate, ssd_algorithm=self.model_algorithm))

    def set_optimizer(self):
        pass

    def set_lr_scheduler(self):
        pass

    def set_criterion(self):
        pass
