import torch
import numpy as np

from configs.centernet_cfg import Config
from lib.loss.centernet_loss import CombinedLoss, RegL1Loss
from lib.models import CenterNet
from lib.utils.bboxes import xywh_to_xyxy, truncate_array, xywh_to_xyxy_torch
from lib.utils.gaussian import gaussian_radius, draw_umich_gaussian
from lib.utils.image_process import read_image_and_convert_to_tensor, read_image, reverse_letter_box
from lib.utils.nms import diou_nms
from lib.utils.visualize import show_detection_results


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

        self.K = cfg.decode.max_boxes_per_img
        self.conf_threshold = cfg.decode.score_threshold
        self.nms_threshold = cfg.decode.nms_threshold
        self.use_nms = cfg.decode.use_nms

        self.letterbox_image = cfg.decode.letterbox_image

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
        class_label = label[:, 1:2]
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

    def predict(self, model, image_path, print_on, save_result):
        model.eval()
        # 处理单张图片
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=self.letterbox_image)
        image = image.to(self.device)

        with torch.no_grad():
            preds = model(image)
            boxes, scores, classes = self.decode_boxes(preds, h, w)

        if boxes.shape[0] == 0:
            print(f"No object detected")
            return read_image(image_path, mode='bgr')

        return show_detection_results(image_path=image_path,
                                      dataset_name=self.cfg.dataset.dataset_name,
                                      boxes=boxes,
                                      scores=scores,
                                      class_indices=classes,
                                      print_on=print_on,
                                      save_result=save_result,
                                      save_dir=self.cfg.decode.test_results)

    def decode_boxes(self, pred, h, w, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        heatmap = pred[..., :self.num_classes]
        reg = pred[..., self.num_classes: self.num_classes + 2]
        wh = pred[..., -2:]
        batch_size = heatmap.size(0)

        heatmap = torch.sigmoid(heatmap)
        heatmap = CenterNetA._suppress_redundant_centers(heatmap)
        scores, inds, classes, ys, xs = CenterNetA._top_k(scores=heatmap, k=self.K)
        if reg is not None:
            reg = RegL1Loss.gather_feat(feat=reg, ind=inds)
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + reg[:, :, 0]  # shape: (batch_size, self.K)
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + reg[:, :, 1]
        else:
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + 0.5
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + 0.5
        wh = RegL1Loss.gather_feat(feat=wh, ind=inds)  # shape: (batch_size, self.K, 2)
        classes = torch.reshape(classes, (batch_size, self.K))
        scores = torch.reshape(scores, (batch_size, self.K))

        bboxes = torch.cat(tensors=[xs.unsqueeze(-1), ys.unsqueeze(-1), wh], dim=-1)  # shape: (batch_size, self.K, 4)
        bboxes[..., ::2] /= self.feature_size[1]
        bboxes[..., 1::2] /= self.feature_size[0]
        bboxes = torch.clamp(bboxes, min=0, max=1)
        # (cx, cy, w, h) ----> (xmin, ymin, xmax, ymax)
        bboxes = xywh_to_xyxy_torch(bboxes)

        score_mask = scores >= conf_threshold  # shape: (batch_size, self.K)

        bboxes = bboxes[score_mask]
        scores = scores[score_mask]
        classes = classes[score_mask]
        if self.use_nms:
            indices = diou_nms(boxes=bboxes, scores=scores, iou_threshold=self.nms_threshold)
            bboxes, scores, classes = bboxes[indices], scores[indices], classes[indices]

        boxes = reverse_letter_box(h=h, w=w, input_size=self.input_size, boxes=bboxes, xywh=False)
        # 转化为numpy.ndarray
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        return boxes, scores, classes

    @staticmethod
    def _suppress_redundant_centers(heatmap, pool_size=3):
        """
        消除8邻域内的其它峰值点
        :param heatmap:
        :param pool_size:
        :return:
        """
        hmax = torch.nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=((pool_size - 1) // 2))(heatmap)
        keep = torch.eq(heatmap, hmax).to(torch.float32)
        return heatmap * keep

    @staticmethod
    def _top_k(scores, k):
        B, H, W, C = scores.size()
        scores = torch.reshape(scores, shape=(B, -1))
        topk_scores, topk_inds = torch.topk(input=scores, k=k, largest=True, sorted=True)
        topk_clses = topk_inds % C  # 应该选取哪些通道（类别）
        pixel = torch.div(topk_inds, C, rounding_mode="floor")
        topk_ys = torch.div(pixel, W, rounding_mode="floor")  # 中心点的y坐标
        topk_xs = pixel % W  # 中心点的x坐标
        topk_inds = (topk_ys * W + topk_xs).to(torch.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
