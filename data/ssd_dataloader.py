import torch

from data.dataloader import PublicDataLoader
from utils.anchor import generate_ssd_anchor
from utils.bboxes import jaccard
from configs.ssd import Config


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
        self.val_transforms.append(UpdateClassIndices())
        self.val_transforms.append(AssignGTToDefaultBoxes(cfg, anchors))
