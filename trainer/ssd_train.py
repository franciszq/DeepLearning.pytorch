import os
import traceback
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.ssd import Config
from data.ssd_dataloader import SSDLoader
from loss.multi_box_loss import MultiBoxLoss
from models.ssd import SSD
from trainer.base import Pipeline
from utils.anchor import generate_ssd_anchor
from utils.ckpt import CheckPoint
from utils.metrics import MeanMetric


class SSDTrainer(Pipeline):
    def __init__(self, cfg: Config, device):
        self.device = device
        self.cfg = cfg
        self.dataset_name = cfg.dataset.dataset_name
        self.batch_size = cfg.train.batch_size
        self.input_image_size = cfg.arch.input_size
        self.resume_training_weights = cfg.train.resume_training
        self.optimizer_name = cfg.optimizer.name
        self.total_epoch = cfg.train.epoch
        self.num_workers = cfg.train.num_workers
        self.test_images_dir = cfg.decode.test_images_dir
        self.eval_interval = cfg.train.eval_interval
        self.save_interval = cfg.train.save_interval
        self.save_path = cfg.train.save_path
        # 自动创建目录用于存放保存的模型
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.result_path = cfg.decode.test_results
        # 自动创建目录用于存放姿态估计的结果
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.tensorboard_on = cfg.train.tensorboard_on
        self.mixed_precision = cfg.train.mixed_precision

        # 生成anchor
        self.anchors = generate_ssd_anchor(input_image_shape=self.input_image_size[1:],
                                           anchor_sizes=cfg.arch.anchor_size,
                                           feature_shapes=cfg.arch.feature_shapes,
                                           aspect_ratios=cfg.arch.aspect_ratios)  # (8732, 4)

        self.train_dataloader = None
        self.val_dataloader = None
        self.model = None

        # 加载数据集
        self._load_data()
        # 创建网络模型
        self._initialize_model()

        # 损失函数
        self.criterion = MultiBoxLoss(self.anchors.copy(),
                                      cfg.loss.overlap_threshold,
                                      cfg.loss.variance,
                                      cfg.loss.neg_pos,
                                      self.device)

        # 创建优化器
        if self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            raise ValueError(f"不支持{self.optimizer_name}优化器")

    def _load_data(self, *args, **kwargs):
        self.train_dataloader, self.val_dataloader = SSDLoader(self.cfg, self.dataset_name, self.batch_size,
                                                               self.input_image_size[1:], self.anchors.copy()).__call__()

    def _initialize_model(self, *args, **kwargs):
        self.model = SSD(self.cfg)
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
        loc_loss_mean = MeanMetric()  # 位置损失
        conf_loss_mean = MeanMetric()  # 分类损失
        start_epoch = 0

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
            print(f"成功加载权重文件{self.resume_training_weights}！")

        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(start_epoch + 1, self.total_epoch + 1):
            # 切换为训练模式
            self.model.train()
            # 重置
            loss_mean.reset()
            loc_loss_mean.reset()
            conf_loss_mean.reset()

            with tqdm(self.train_dataloader, desc=f"Epoch-{epoch}/{self.total_epoch}") as pbar:
                for i, (images, targets) in enumerate(pbar):
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            preds = self.model(images)
                            total_loss, l_loss, c_loss = self.criterion(y_true=targets, y_pred=preds)
                        scaler.scale(total_loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        preds = self.model(images)
                        total_loss, l_loss, c_loss = self.criterion(y_true=targets, y_pred=preds)
                        total_loss.backward()
                        self.optimizer.step()

                    loss_mean.update(total_loss.item())
                    loc_loss_mean.update(l_loss.item())
                    conf_loss_mean.update(c_loss.item())

                    if self.tensorboard_on:
                        writer.add_scalar(tag="Loss", scalar_value=loss_mean.result(),
                                          global_step=(epoch - 1) * len(self.train_dataloader) + i)
                        writer.add_scalar(tag="Loc Loss", scalar_value=loc_loss_mean.result(),
                                          global_step=(epoch - 1) * len(self.train_dataloader) + i),
                        writer.add_scalar(tag="Conf Loss", scalar_value=conf_loss_mean.result(),
                                          global_step=(epoch - 1) * len(self.train_dataloader) + i)

                    pbar.set_postfix({"loss": "{}".format(loss_mean.result()),
                                      "loc_loss": "{:.4f}".format(loc_loss_mean.result()),
                                      "conf_loss": "{:.4f}".format(conf_loss_mean.result())})

            if epoch % self.eval_interval == 0:
                pass
                # evaluation = self.evaluate(self.val_dataloader)
                # print("Evaluation: loss={loss}, acc={acc}".format(**evaluation))

            if epoch % self.save_interval == 0:
                CheckPoint.save(self.model, self.optimizer, None, epoch,
                                path=Path(self.save_path).joinpath(
                                    f"ssd_{self.dataset_name.lower()}_epoch-{epoch}.pth"))

        if self.tensorboard_on:
            writer.close()
        # 保存最终模型
        CheckPoint.save(self.model, self.optimizer, None, epoch,
                        path=Path(self.save_path).joinpath(f"ssd_{self.dataset_name.lower()}_final.pth"))

    def evaluate(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
