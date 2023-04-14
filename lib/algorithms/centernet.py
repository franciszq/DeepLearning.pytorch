import torch
import numpy as np

from configs.centernet_cfg import Config
from lib.loss.centernet_loss import CombinedLoss
from lib.models import CenterNet
from lib.utils.bboxes import xywh_to_xyxy, truncate_array
from lib.utils.gaussian import gaussian_radius, draw_umich_gaussian


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

        # 输入图片的尺寸
        self.input_size = cfg.arch.input_size[1:]
        # 特征图的下采样倍数
        self.downsampling_ratio = cfg.arch.downsampling_ratio
        # 特征图的尺寸 [h, w]
        self.feature_size = [self.input_size[0] // self.downsampling_ratio,
                             self.input_size[1] // self.downsampling_ratio]

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
        # 确保label的第一个维度是max_num_boxes
        label = truncate_array(label, self.max_num_boxes, False)
        hm = np.zeros((self.feature_size[0], self.feature_size[1], self.num_classes), dtype=np.float32)
        reg = np.zeros((self.max_num_boxes, 2), dtype=np.float32)
        wh = np.zeros((self.max_num_boxes, 2), dtype=np.float32)
        reg_mask = np.zeros((self.max_num_boxes,), dtype=np.float32)
        ind = np.zeros((self.max_num_boxes,), dtype=np.float32)

        for j, item in enumerate(label):
            # 坐标映射到特征图尺寸上
            item[:4:2] = item[:4:2] * self.feature_size[1]
            item[1:4:2] = item[1:4:2] * self.feature_size[0]
            xmin, ymin, xmax, ymax, class_id = item
            # 类别id
            class_id = class_id.astype(np.int32)
            # 目标框的宽高
            h, w = int(ymax - ymin), int(xmax - xmin)
            # 高斯半径
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))
            # 中心点坐标
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = np.array([ctr_x, ctr_y], dtype=np.float32)
            center_point_int = center_point.astype(np.int32)
            _hm = draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)
            hm[:, :, class_id] = _hm

            reg[j] = center_point - center_point_int
            wh[j] = np.array([w, h], dtype=np.float32)
            reg_mask[j] = 1
            ind[j] = center_point_int[1] * self.feature_size[1] + center_point_int[0]

        # 返回torch.Tensor
        return torch.from_numpy(hm), torch.from_numpy(reg), torch.from_numpy(wh), torch.from_numpy(
            reg_mask), torch.from_numpy(ind)
