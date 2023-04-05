import numpy as np
import torch
from configs.ssd_cfg import Config
from lib.loss.multi_box_loss import MultiBoxLossV2
from lib.models.ssd_model import SSD
from lib.utils.bboxes import xywh_to_xyxy


class Ssd:

    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device
        # 输入图片的尺寸
        self.input_image_size = self.cfg.arch.input_size[1:]
        # 与锚框有关的参数
        self.anchor_sizes = self.cfg.arch.anchor_sizes
        self.feature_shapes = self.cfg.arch.feature_shapes
        self.aspect_ratios = self.cfg.arch.aspect_ratios
        # 锚框
        self.anchors = self._get_ssd_anchors()
        self.num_anchors = self.anchors.shape[0]
        # 类别数目
        self.num_classes = self.cfg.dataset.num_classes
        # 正负样本比例
        self.neg_pos_ratio = self.cfg.loss.neg_pos
        variance = np.array(self.cfg.loss.variance, dtype=np.float32)
        # 将variance变成[0.1, 0.1, 0.2, 0.2]
        self.variance = np.repeat(variance, 2, axis=0)
        self.overlap_threshold = self.cfg.loss.overlap_threshold

    def build_model(self):
        """构建网络模型"""
        model = SSD(self.cfg)
        model_name = "SSD"
        return model, model_name

    def build_loss(self):
        """构建损失函数"""
        loss = MultiBoxLossV2(neg_pos_ratio=self.neg_pos_ratio,
                              num_classes=self.num_classes)
        return loss

    def generate_targets(self, label):
        """
        将一个Nx6的Tensor变成一个8732x5的Tensor
        :param label: numpy.ndarray, shape: (N, 6(_, class_id, cx, cy, w, h))
        :return: torch.Tensor, shape: (8732, 5(center_x, center_y, w, h, label))
        """
        # 数据集的类别标签增加1，因为背景的类别是0
        label[:, 1] += 1
        class_label = label[:, 1].astype(np.int32)
        # 坐标 (N, 4)
        coord_label = label[:, 2:]
        # 坐标由(cx, cy, w, h)转换为(xmin, ymin, xmax, ymax)
        coord_label = xywh_to_xyxy(coord_label)
        # one-hot编码  (N, 21)
        one_hot_label = np.eye(self.num_classes + 1)[class_label]
        # 包含坐标和one-hot编码的标签的label (N, 4 + 21)
        true_label = np.concatenate((coord_label, one_hot_label), axis=-1)

        # assignment[:, :4] 坐标
        # assignment[:, 4:-1] one-hot编码
        # assignment[:, -1] 当前先验框是否有对应的目标，0为没有，1为有
        assignment = np.zeros((self.num_anchors, 4 + 1 + self.num_classes + 1),
                              dtype=np.float32)
        assignment[:, 4] = 1.0  # 默认先验框为背景
        if len(true_label) == 0:
            return torch.from_numpy(assignment)
        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self._encode_box, 1,
                                            true_label[:, :4])

        # ---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        # ---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # ---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        # ---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # ---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        # ---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # ---------------------------------------------------#
        #   编码后的真实框的赋值
        # ---------------------------------------------------#
        assignment[:, :4][best_iou_mask] = encoded_boxes[
            best_iou_idx, np.arange(assign_num), :4]
        # ----------------------------------------------------------#
        #   4代表为背景的概率，设为0，因为这些先验框有对应的物体
        # ----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = true_label[best_iou_idx, 5:]
        # ----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        # ----------------------------------------------------------#
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return torch.from_numpy(assignment)

    def _encode_box(self, box, return_iou=True):
        # ---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        # ---------------------------------------------#

        # ---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        # ---------------------------------------------#
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # ---------------------------------------------#
        #   真实框的面积
        # ---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # ---------------------------------------------#
        #   先验框的面积
        # ---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (
            self.anchors[:, 3] - self.anchors[:, 1])
        # ---------------------------------------------#
        #   计算iou
        # ---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union

        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        # ---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        # ---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        # ---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        # ---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # ---------------------------------------------#
        #   利用iou进行赋值
        # ---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # ---------------------------------------------#
        #   找到对应的先验框
        # ---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        # ---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        # ---------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_anchors_center = (assigned_anchors[:, 0:2] +
                                   assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] -
                               assigned_anchors[:, 0:2])

        # ------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(self.variance)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(self.variance)[2:4]
        return encoded_box.ravel()

    def _get_ssd_anchors(self):
        image_h, image_w = self.input_image_size
        anchors = []
        for i in range(len(self.feature_shapes)):
            # 先验框的短边和长边
            min_size = self.anchor_sizes[i]
            max_size = self.anchor_sizes[i + 1]
            # 特征图的高和宽，它们相等
            feature_h = self.feature_shapes[i]
            # 对于每个像素位置，根据aspect_ratio生成不同宽、高比的先验框
            box_widths = []
            box_heights = []
            for ar in self.aspect_ratios[i]:
                if ar == 1:
                    box_widths.append(min_size)
                    box_heights.append(min_size)
                    box_widths.append(np.sqrt(min_size * max_size))
                    box_heights.append(np.sqrt(min_size * max_size))
                else:
                    box_widths.append(min_size * np.sqrt(ar))
                    box_heights.append(min_size / np.sqrt(ar))

            half_box_widths = np.array(
                box_widths) / 2.0  # shape: (len(aspect_ratios[i])+1,)
            half_box_heights = np.array(box_heights) / 2.0

            # 特征层上一个像素点映射到原图上对应的像素长度
            pixel_length = [image_h / feature_h, image_w / feature_h]
            # 生成网格中心
            c_x = np.linspace(0.5 * pixel_length[1],
                              image_w - 0.5 * pixel_length[1], feature_h)
            c_y = np.linspace(0.5 * pixel_length[0],
                              image_h - 0.5 * pixel_length[0], feature_h)
            center_x, center_y = np.meshgrid(c_x, c_y)
            center_x = np.reshape(center_x, (-1, 1))  # (feature_h**2, 1)
            center_y = np.reshape(center_y, (-1, 1))  # (feature_h**2, 1)

            anchor = np.concatenate((center_x, center_y),
                                    axis=1)  # (feature_h**2, 2)
            # 对于每一种宽高比例，都需要一个对应的先验框
            # shape: (feature_h**2, 4*(len(aspect_ratios[i])+1))
            anchor = np.tile(anchor, (1, (len(self.aspect_ratios[i]) + 1) * 2))

            # 转换为xmin, ymin, xmax, ymax格式
            # shape: (feature_h**2, len(aspect_ratios[i])+1)
            anchor[:, ::4] -= half_box_widths
            anchor[:, 1::4] -= half_box_heights
            anchor[:, 2::4] += half_box_widths
            anchor[:, 3::4] += half_box_heights

            # 归一化
            anchor[:, ::2] /= image_w
            anchor[:, 1::2] /= image_h
            anchor = np.clip(anchor, a_min=0.0, a_max=1.0)
            anchor = np.reshape(anchor, (-1, 4))

            anchors.append(anchor)

        anchors = np.concatenate(anchors, axis=0)  # (8732, 4)
        return anchors.astype(dtype=np.float32)
