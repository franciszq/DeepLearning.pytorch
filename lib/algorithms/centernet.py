import numpy as np

from configs.centernet_cfg import Config
from lib.loss.centernet_loss import CombinedLoss
from lib.models import CenterNet
from lib.utils.bboxes import xywh_to_xyxy, truncate_array


class CenterNetA:
    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device

        # 类别数目
        self.num_classes = cfg.dataset.num_classes

        # 损失函数中的权重分配
        self.hm_weight = cfg.loss.hm_weight
        self.wh_weight = cfg.loss.wh_weight
        self.off_weight = cfg.loss.off_weight

        # 每张图片中最多的目标数目
        self.max_num_boxes = cfg.train.max_num_boxes

        

    def build_model(self):
        model = CenterNet(self.cfg)
        model_name = "CenterNet"
        return model, model_name

    def build_loss(self):
        return CombinedLoss(self.num_classes, self.hm_weight, self.wh_weight, self.off_weight)

    def generate_targets(self, label):
        """
        :param label: numpy.ndarray, shape: (N, 6(_, class_id, cx, cy, w, h))
        :return:
        """
        class_label = label[:, 1]
        # 坐标由(cx, cy, w, h)转换为(xmin, ymin, xmax, ymax)
        coord_label = xywh_to_xyxy(label[:, 2:])
        # shape: (N, 5(xmin, ymin, xmax, ymax, class_id))
        label = np.concatenate((coord_label, class_label), axis=-1)
        # 确保label的第一个维度是max_num_boxes，不足的用-1填充
        label = truncate_array(label, self.max_num_boxes)
        

