import numpy as np
import torch


def yolo7_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i   # 图片在这个batch中的编号
        bboxes.append(box)
    images = torch.stack(images, dim=0)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes


def ssd_collate(batch, ssd_algorithm):
    images = []
    targets = []
    for i, (image, target) in enumerate(batch):
        images.append(image)
        target = ssd_algorithm.generate_targets(target)
        targets.append(target)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)
    return images, targets


def centernet_collate(batch, centernet_algorithm):
    images = []
    targets = []
    # gt_heatmap = []
    # gt_reg = []
    # gt_wh = []
    # gt_reg_mask = []
    # gt_indices = []
    for i, (image, label) in enumerate(batch):
        images.append(image)
        heatmap, reg, wh, reg_mask, indices = centernet_algorithm.generate_targets(label)
        targets.append([heatmap, reg, wh, reg_mask, indices])
        # gt_heatmap.append(heatmap)
        # gt_reg.append(reg)
        # gt_wh.append(wh)
        # gt_reg_mask.append(reg_mask)
        # gt_indices.append(indices)
    targets = [torch.stack(t, dim=0) for t in zip(*targets)]
    # images = torch.stack(images, dim=0)
    # gt_heatmap = torch.stack(gt_heatmap, dim=0)
    # gt_reg = torch.stack(gt_reg, dim=0)
    # gt_wh = torch.stack(gt_wh, dim=0)
    # gt_reg_mask = torch.stack(gt_reg_mask, dim=0)
    # gt_indices = torch.stack(gt_indices, dim=0)
    return images, targets
