import torch
import torch.nn.functional as F

from utils.bboxes import xyxy_to_xywh


class MultiBoxLoss:
    def __init__(self, anchors, threshold, variance, negpos_ratio, device):
        self.device = device
        # torch.Tensor, shape: (先验框总数(8732), 4)
        self.default_boxes = torch.from_numpy(xyxy_to_xywh(anchors)).to(device)
        self.default_boxes.requires_grad = False
        self.default_boxes = self.default_boxes.unsqueeze(dim=0)  # shape: (1, 8732, 4)
        self.threshold = threshold
        self.variance = variance
        self.negpos_ratio = negpos_ratio
        self.scale_xy = 1.0 / self.variance[0]
        self.scale_wh = 1.0 / self.variance[1]

    def _location_vec(self, loc):

        g_cxcy = self.scale_xy * (loc[..., :2] - self.default_boxes[..., :2]) / self.default_boxes[..., 2:]
        g_wh = self.scale_wh * torch.log(loc[..., 2:] / self.default_boxes[..., 2:])
        return torch.cat(tensors=(g_cxcy, g_wh), dim=-1)

    def __call__(self, y_true, y_pred):
        """
        :param y_true: torch.Tensor, shape: (batch_size, 8732, 5(cx, cy, w, h, class_index))
        :param y_pred: (loc, conf), 其中loc的shape是(batch_size, 8732, 4), conf的shape是(batch_size, 8732, self.num_classes)
        :return:
        """
        ploc, plabel = y_pred
        gloc = y_true[..., :-1]  # (batch_size, 8732, 4)
        glabel = y_true[..., -1].long()  # (batch_size, 8732)

        # 筛选正样本
        mask = glabel > 0  # (batch_size, 8732)
        # 正样本个数
        pos_num = mask.sum(dim=1)  # (batch_size)

        # 偏移量
        vec_gd = self._location_vec(gloc)  # (batch_size, 8732, 4)
        # 位置损失
        loc_loss = F.smooth_l1_loss(ploc, vec_gd, reduction="none").sum(dim=-1)  # (batch_size, 8732)
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # (batch_size)

        con = F.cross_entropy(torch.permute(plabel, dims=(0, 2, 1)), glabel, reduction="none")  # (batch_size, 8732)

        # Hard Negative Mining
        con_neg = con.clone()
        con_neg[mask] = torch.tensor(0.0)
        # 排序，得到一个索引，它的值表示这个位置的元素第几大
        _, con_idx = con_neg.sort(1, descending=True)
        _, con_rank = con_idx.sort(1)
        neg_num = torch.clamp(self.negpos_ratio * pos_num, max=mask.size(1)).unsqueeze(1)  # (batch_size, 1)
        neg_mask = con_rank < neg_num  # (batch_size, 8732)

        # 分类损失
        con_loss = (con * (mask.float() + neg_mask.float())).sum(1)  # (batch_size)

        total_loss = loc_loss + con_loss
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        total_loss = (total_loss * num_mask / pos_num).mean(dim=0)
        loss_l = (loc_loss * num_mask / pos_num).mean(dim=0)
        loss_c = (con_loss * num_mask / pos_num).mean(dim=0)
        return total_loss, loss_l, loss_c