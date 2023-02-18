import os
import traceback
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.ckpt import CheckPoint
from utils.metrics import MeanMetric


class Pipeline(metaclass=ABCMeta):

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def _initialize_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


class BaseTrainer:
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        # 模型总的训练轮数
        self.total_epoch = cfg.train.epoch
        # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
        self.last_epoch = cfg.train.last_epoch
        # 恢复训练时加载的checkpoint文件
        self.resume_training_weights = cfg.train.resume_training
        # 数据集名称
        self.dataset_name = cfg.dataset.dataset_name
        # 数据集中的目标类别数，对于voc，这个值是20，对于coco，它是80
        self.num_classes = cfg.arch.num_classes
        # 模型输入图片大小 (C, H, W)
        self.input_image_size = cfg.arch.input_size
        self.batch_size = cfg.train.batch_size
        # 优化器
        self.optimizer_name = cfg.optimizer.name
        self.warmup_epochs = cfg.train.warmup_epochs
        self.initial_lr = cfg.train.initial_lr
        self.milestones = cfg.train.milestones
        self.gamma = cfg.train.gamma
        self.num_workers = cfg.train.num_workers
        self.eval_interval = cfg.train.eval_interval
        self.save_interval = cfg.train.save_interval
        self.save_path = cfg.train.save_path
        # 自动创建目录用于存放保存的模型
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.result_path = cfg.decode.test_results
        # 自动创建目录用于存放检测结果
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.tensorboard_on = cfg.train.tensorboard_on
        self.mixed_precision = cfg.train.mixed_precision
        self.writer = None

        # 训练集
        self.train_dataloader = None
        # 模型
        self.model = None

        # 优化器
        self.optimizer = None
        # 学习率调整策略
        self.lr_scheduler = None
        # 损失函数
        self.criterion = None

        self.metric_names = []

        self.load_data()
        self.initialize_model()
        self.set_optimizer()
        self.set_lr_scheduler()
        self.set_criterion()

    def load_weights(self, weights=None):
        if weights:
            return CheckPoint.load(path=weights, device=self.device,
                                   model=self.model, optimizer=self.optimizer)
        else:
            return CheckPoint.load(path=self.resume_training_weights, device=self.device,
                                   model=self.model, optimizer=self.optimizer)

    def initialize_model(self):
        pass

    def load_data(self):
        pass

    def set_optimizer(self):
        pass

    def set_lr_scheduler(self):
        pass

    def set_criterion(self):
        pass

    def train_loop(self, images, targets, scaler) -> List:
        return []

    def train(self):
        n = len(self.metric_names)
        train_metrics = [MeanMetric() for _ in range(n)]
        if self.tensorboard_on:
            writer = SummaryWriter()
            try:
                writer.add_graph(self.model, torch.randn(self.batch_size, *self.input_image_size, dtype=torch.float32,
                                                         device=self.device))
            except Exception:
                traceback.print_exc()

        if self.resume_training_weights != "":
            # 从checkpoint恢复训练
            _, _, _, start_epoch = self.load_weights()
            assert self.last_epoch == start_epoch, f"last epoch should be {start_epoch}, but got {self.last_epoch}"
            print(f"After loading weights from {self.resume_training_weights}, it will resume training soon.")

        scaler = None
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.last_epoch + 1, self.total_epoch):
            # 切换为训练模式
            self.model.train()
            # 重置
            for m in train_metrics:
                m.reset()
            with tqdm(self.train_dataloader, desc=f"Epoch-{epoch}/{self.total_epoch}") as pbar:
                for i, (images, targets) in enumerate(pbar):
                    metrics = self.train_loop(images, targets, scaler)
                    assert len(metrics) == n
                    for j in range(n):
                        train_metrics[j].update(metrics[j].item())
                    # 当前学习率
                    current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    if self.tensorboard_on:
                        for k in range(n):
                            writer.add_scalar(tag=f"Train/{self.metric_names[k]}",
                                              scalar_value=metrics[k].item(),
                                              global_step=epoch * len(self.train_dataloader) + i)

                    # 设置进度条后缀
                    postfix_info = {}
                    for p in range(n):
                        postfix_info[self.metric_names[p]] = f"{(train_metrics[p].result()):.5f}"
                    pbar.set_postfix(postfix_info)

            self.lr_scheduler.step()

            if epoch % self.save_interval == 0:
                CheckPoint.save(self.model, self.optimizer, None, epoch,
                                path=Path(self.save_path).joinpath(
                                    f"{self.model.get_model_name()}_{self.dataset_name.lower()}_epoch-{epoch}.pth"))
        if self.tensorboard_on:
            writer.close()
        # 保存最终模型
        CheckPoint.save(self.model, self.optimizer, None, self.total_epoch - 1,
                        path=Path(self.save_path).joinpath(f"{self.model.get_model_name()}_{self.dataset_name.lower()}_final.pth"))
