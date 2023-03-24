import numpy as np
import torch

from lib.data.dataloader import PublicDataLoader
from lib.utils.bboxes import jaccard
from configs.ssd_cfg import Config
import torchvision.transforms.functional as F


class UpdateClassIndices:
    def __call__(self, image, target):
        target[..., -1] += 1  # 背景的label是0
        return image, target


class AssignGTToDefaultBoxes:
    def __init__(self, cfg: Config, anchors):
        self.default_boxes = anchors  # shape: (8732, 4)
        # To tensor
        self.default_boxes = torch.from_numpy(self.default_boxes)
        self.threshold = cfg.loss.overlap_threshold

    def __call__(self, image, target):
        """
        将一个Nx5的Tensor变成一个8732x5的Tensor
        :param image: torch.Tensor, 图片
        :param target: torch.Tensor, shape: (N, 5(xmin, ymin, xmax, ymax, label))
        :return: image, target_out [shape: (8732, 5(center_x, center_y, w, h, label))]
        """
        boxes = target[:, :-1]
        labels_in = target[:, -1].long()
        overlaps = jaccard(boxes, self.default_boxes)
        # 每个default_box对应的最大IoU值的gt_box
        best_dbox_ious, best_dbox_idx = overlaps.max(dim=0)  # [8732]
        # 每个gt_box对应的最大IoU值的default_box
        best_bbox_ious, best_bbox_idx = overlaps.max(dim=1)  # [N]

        # 将每个gt匹配到的最佳default_box设置为正样本
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        # 将相应default box匹配最大iou的gt索引进行替换
        idx = torch.arange(0, best_bbox_idx.size(dim=0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # 将与gt的IoU大于给定阈值的default_boxes设置为正样本
        masks = best_dbox_ious > self.threshold
        labels_out = torch.zeros(self.default_boxes.size(0), dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        # 将default_box匹配到正样本的位置设置成对应的gt信息
        bboxes_out = self.default_boxes.clone()
        bboxes_out[masks, :] = boxes[best_dbox_idx[masks], :]

        cx = (bboxes_out[:, 0] + bboxes_out[:, 2]) / 2
        cy = (bboxes_out[:, 1] + bboxes_out[:, 3]) / 2
        w = bboxes_out[:, 2] - bboxes_out[:, 0]
        h = bboxes_out[:, 3] - bboxes_out[:, 1]
        bboxes_out[:, 0] = cx
        bboxes_out[:, 1] = cy
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h

        target_out = torch.cat(tensors=(bboxes_out, labels_out.unsqueeze(-1)), dim=-1)

        return image, target_out


class SSDLoader(PublicDataLoader):
    def __init__(self, cfg: Config, dataset_name: str, batch_size, input_size, anchors):
        super().__init__(dataset_name, batch_size, input_size)
        self.train_transforms.append(UpdateClassIndices())
        self.train_transforms.append(AssignGTToDefaultBoxes(cfg, anchors))
        # self.val_transforms.append(UpdateClassIndices())
        # self.val_transforms.append(AssignGTToDefaultBoxes(cfg, anchors))


class AssignGTToDefaultBoxesV2:
    def __init__(self, cfg: Config, anchors):
        self.anchors = anchors  # shape: (8732, 4)
        self.num_anchors = anchors.shape[0]
        self.overlap_threshold = cfg.loss.overlap_threshold
        self.num_classes = cfg.arch.num_classes + 1

    def __call__(self, image, target):
        """
        将一个Nx5的Tensor变成一个8732x5的Tensor
        :param image: numpy.ndarray, 图片
        :param target: numpy.ndarray, shape: (N, 5(xmin, ymin, xmax, ymax, label))
        :return: image, target_out [shape: (8732, 5(center_x, center_y, w, h, label))]
        """
        image_data = F.to_tensor(image)
        coord_label = target[:, :4].copy()
        target = target.astype(np.int32)
        one_hot_label = np.eye(self.num_classes - 1)[target[:, 4]]
        box = np.concatenate([coord_label, one_hot_label], axis=-1)
        box = self._assign_boxes(box)
        return image_data, torch.from_numpy(box)

    def _assign_boxes(self, boxes):
        # ---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        # ---------------------------------------------------#
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self._encode_box, 1, boxes[:, :4])
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
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # ----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        # ----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        # ----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        # ----------------------------------------------------------#
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def _iou(self, box):
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
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        # ---------------------------------------------#
        #   计算iou
        # ---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def _encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):
        # ---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        # ---------------------------------------------#
        iou = self._iou(box)
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
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])

        # ------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()


class SSDLoaderV2(PublicDataLoader):
    def __init__(self, cfg: Config, dataset_name: str, batch_size, input_size, anchors):
        super().__init__(dataset_name, batch_size, input_size)
        # 移除ToTensor()变换
        self.train_transforms.pop(1)
        self.train_transforms.insert(1, AssignGTToDefaultBoxesV2(cfg, anchors))
