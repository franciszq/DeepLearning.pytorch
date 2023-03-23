import os
import traceback
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.ssd_cfg import Config
from data.ssd_dataloader import SSDLoaderV2
from loss.multi_box_loss import MultiBoxLossV2
from metrics.eval import evaluate_pipeline
from models.ssd_model import SSD
from predict.ssd_decode import DecoderV2
from trainer.base import Pipeline, MeanMetric
from utils.anchor import generate_ssd_anchor_v2
from utils.ckpt import CheckPoint
from trainer.lr_scheduler import warm_up_scheduler


class SSDTrainer(Pipeline):
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

        # 生成anchor
        self.anchors = generate_ssd_anchor_v2(input_image_shape=self.input_image_size[1:],
                                              anchor_sizes=cfg.arch.anchor_size,
                                              feature_shapes=cfg.arch.feature_shapes,
                                              aspect_ratios=cfg.arch.aspect_ratios)  # (8732, 4)

        self.train_dataloader = None
        # self.val_dataloader = None
        self.model = None

        # 加载数据集
        self._load_data()
        # 创建网络模型
        self._initialize_model()
        self.model_name = self.model.get_model_name()

        # 损失函数
        self.criterion = MultiBoxLossV2(neg_pos_ratio=cfg.loss.neg_pos,
                                        num_classes=cfg.arch.num_classes)

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
        self.train_dataloader = SSDLoaderV2(self.cfg, self.dataset_name, self.batch_size,
                                            self.input_image_size[1:],
                                            self.anchors.copy()).__call__()

    def _initialize_model(self, *args, **kwargs):
        self.model = SSD(self.cfg)
        self.model.to(device=self.device)

    def train(self, *args, **kwargs):
        loss_mean = MeanMetric()
        loc_loss_mean = MeanMetric()  # 位置损失
        conf_loss_mean = MeanMetric()  # 分类损失

        if self.tensorboard_on:
            writer = SummaryWriter()
            try:
                writer.add_graph(self.model, torch.randn(self.batch_size, *self.input_image_size, dtype=torch.float32,
                                                         device=self.device))
            except Exception:
                traceback.print_exc()

        if self.resume_training_weights != "":
            # 从checkpoint恢复训练
            CheckPoint.load(path=self.resume_training_weights,
                            device=self.device,
                            model=self.model,
                            pure=False,
                            optimizer=self.optimizer,
                            scheduler=self.lr_scheduler)
            print(f"After loading weights from {self.resume_training_weights}, it will resume training soon.")

        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.last_epoch + 1, self.total_epoch):
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

                    # 当前学习率
                    current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                    if self.tensorboard_on:
                        writer.add_scalar(tag="Train/loss", scalar_value=total_loss.item(),
                                          global_step=epoch * len(self.train_dataloader) + i)
                        writer.add_scalar(tag="Train/loc loss", scalar_value=l_loss.item(),
                                          global_step=epoch * len(self.train_dataloader) + i),
                        writer.add_scalar(tag="Train/conf loss", scalar_value=c_loss.item(),
                                          global_step=epoch * len(self.train_dataloader) + i)
                        writer.add_scalar(tag="Learning rate", scalar_value=current_lr,
                                          global_step=epoch * len(self.train_dataloader) + i)

                    pbar.set_postfix({"loss": "{}".format(loss_mean.result()),
                                      "loc_loss": "{:.4f}".format(loc_loss_mean.result()),
                                      "conf_loss": "{:.4f}".format(conf_loss_mean.result())})

            self.lr_scheduler.step()

            if epoch % self.save_interval == 0:
                CheckPoint.save(model=self.model, path=Path(self.save_path).joinpath(
                    f"{self.model_name}_{self.dataset_name.lower()}_epoch-{epoch}.pth"),
                                optimizer=self.optimizer,
                                scheduler=self.lr_scheduler)

        if self.tensorboard_on:
            writer.close()
        # 保存最终模型
        CheckPoint.save(model=self.model,
                        path=Path(self.save_path).joinpath(f"{self.model_name}_{self.dataset_name.lower()}_final.pth"))

    def evaluate(self,
                 weights=None,
                 subset='val',
                 skip=False):

        # 加载权重
        if weights is not None:
            CheckPoint.load(path=weights,
                            device=self.device,
                            model=self.model,
                            pure=False)
        # 切换为'eval'模式
        self.model.eval()

        # evaluate_pipeline(model=self.model,
        #                   decoder=Decoder(anchors=self.anchors.copy(),
        #                                   input_image_size=self.input_image_size[1:],
        #                                   num_max_output_boxes=self.cfg.decode.num_max_output_boxes,
        #                                   num_classes=self.cfg.arch.num_classes,
        #                                   variance=self.cfg.loss.variance,
        #                                   conf_threshold=0.02,
        #                                   nms_threshold=self.cfg.decode.nms_threshold,
        #                                   device=self.device),
        #                   input_image_size=self.input_image_size[1:],
        #                   map_out_root=os.path.join(self.result_path, "map"),
        #                   subset=subset,
        #                   device=self.device,
        #                   skip=skip)
        evaluate_pipeline(model=self.model,
                          decoder=DecoderV2(anchors=self.anchors.copy(),
                                            ori_image_shape=[0, 0],
                                            input_image_size=self.input_image_size[1:],
                                            num_max_output_boxes=self.cfg.decode.num_max_output_boxes,
                                            num_classes=self.cfg.arch.num_classes,
                                            variance=self.cfg.loss.variance,
                                            conf_threshold=0.02,
                                            nms_threshold=self.cfg.decode.nms_threshold,
                                            device=self.device),
                          input_image_size=self.input_image_size[1:],
                          map_out_root=os.path.join(self.result_path, "map"),
                          subset=subset,
                          list_input=True,
                          device=self.device,
                          skip=skip)
