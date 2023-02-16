import os
import traceback
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.centernet import Config
from data.centernet_dataloader import CenterNetLoader
from data.centernet_target import TargetGenerator
from loss.centernet_loss import CombinedLoss
from mAP.eval import evaluate_pipeline
from models.centernet import CenterNet
from trainer.base import Pipeline
from utils.ckpt import CheckPoint
from utils.lr_scheduler import warm_up_scheduler
from utils.metrics import MeanMetric


class CenterNetTrainer(Pipeline):
    def __init__(self, cfg: Config, device):
        self.device = device
        self.cfg = cfg
        self.last_epoch = cfg.train.last_epoch
        self.dataset_name = cfg.dataset.dataset_name
        self.batch_size = cfg.train.batch_size

        self.warmup_epochs = cfg.train.warmup_epochs
        self.initial_lr = cfg.train.initial_lr
        self.milestones = cfg.train.milestones
        self.gamma = cfg.train.gamma

        self.input_image_size = cfg.arch.input_size
        self.resume_training_weights = cfg.train.resume_training
        self.optimizer_name = cfg.optimizer.name
        self.total_epoch = cfg.train.epoch
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

        self.train_dataloader = None
        self.model = None

        # 加载数据集
        self._load_data()
        # 创建网络模型
        self._initialize_model()

        # 损失函数
        self.criterion = CombinedLoss(self.cfg)

        # 创建优化器
        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam([{"params": self.model.parameters(),
                                                'initial_lr': self.initial_lr}], lr=self.initial_lr)
        else:
            raise ValueError(f"{self.optimizer_name} is not supported")

        # 设置优化器
        self.lr_scheduler = warm_up_scheduler(self.optimizer, warmup_epochs=self.warmup_epochs,
                                              milestones=self.milestones, gamma=self.gamma,
                                              last_epoch=self.last_epoch)

    def _load_data(self, *args, **kwargs):
        self.train_dataloader = CenterNetLoader(self.cfg, dataset_name=self.dataset_name,
                                                batch_size=self.batch_size,
                                                input_size=self.input_image_size[1:]).__call__()

    def _initialize_model(self, *args, **kwargs):
        self.model = CenterNet(self.cfg)
        self.model.to(device=self.device)

    def load_weights(self, weights=None):
        if weights:
            return CheckPoint.load(path=weights, device=self.device,
                                   model=self.model, optimizer=self.optimizer)
        else:
            return CheckPoint.load(path=self.resume_training_weights, device=self.device,
                                   model=self.model, optimizer=self.optimizer)

    def train(self, *args, **kwargs):
        loss_mean = MeanMetric()

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

        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.last_epoch + 1, self.total_epoch):
            # 切换为训练模式
            self.model.train()
            # 重置
            loss_mean.reset()

            with tqdm(self.train_dataloader, desc=f"Epoch-{epoch}/{self.total_epoch}") as pbar:
                for i, (images, labels) in enumerate(pbar):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    target = list(TargetGenerator(self.cfg, labels, self.device).__call__())

                    self.optimizer.zero_grad()
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            preds = self.model(images)
                            target.insert(0, preds)
                            loss = self.criterion(*target)
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        preds = self.model(images)
                        target.insert(0, preds)
                        loss = self.criterion(*target)
                        loss.backward()
                        self.optimizer.step()

                    loss_mean.update(loss.item())

                    # 当前学习率
                    current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                    if self.tensorboard_on:
                        writer.add_scalar(tag="Train/loss", scalar_value=loss.item(),
                                          global_step=epoch * len(self.train_dataloader) + i)
                        writer.add_scalar(tag="Learning rate", scalar_value=current_lr,
                                          global_step=epoch * len(self.train_dataloader) + i)

                    pbar.set_postfix({"loss": "{}".format(loss_mean.result())})

            self.lr_scheduler.step()

            if epoch % self.save_interval == 0:
                CheckPoint.save(self.model, self.optimizer, None, epoch,
                                path=Path(self.save_path).joinpath(
                                    f"centernet_{self.dataset_name.lower()}_epoch-{epoch}.pth"))

        if self.tensorboard_on:
            writer.close()
        # 保存最终模型
        CheckPoint.save(self.model, self.optimizer, None, self.total_epoch - 1,
                        path=Path(self.save_path).joinpath(f"centernet_{self.dataset_name.lower()}_final.pth"))


    def evaluate(self, *args, **kwargs):
        pass
