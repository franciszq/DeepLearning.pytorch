import torch

from configs.centernet_cfg import Config
from lib.loss.centernet_loss import RegL1Loss
from lib.utils.bboxes import xywh_to_xyxy_torch
from lib.utils.image_process import read_image, letter_box, reverse_letter_box
from lib.utils.nms import diou_nms
import torchvision.transforms.functional as TF

from lib.utils.visualize import show_detection_results


class Decoder:
    def __init__(self, cfg: Config, input_image_size, score_threshold, device):
        """
        初始化参数
        :param cfg:
        :param input_image_size: list or tuple [h, w], CenterNet网络输入图片的固定大小
        :param score_threshold: 分数低于这个数值的目标都会被移除
        :param device: 设备
        """
        self.device = device
        self.K = cfg.decode.max_boxes_per_img
        self.num_classes = cfg.arch.num_classes

        assert input_image_size[0] == input_image_size[1]
        self.input_image_size = input_image_size[0]

        self.downsampling_ratio = cfg.arch.downsampling_ratio
        self.feature_size = self.input_image_size / self.downsampling_ratio
        self.score_threshold = score_threshold
        self.use_nms = cfg.decode.use_nms

    def __call__(self, outputs):
        """
        :param outputs: CenterNet网络输出的feature map
        :return: list of torch.Tensor, shape: [torch.Size([N, 4]) torch.Size([N]) torch.Size([N])]
        """
        heatmap = outputs[..., :self.num_classes]
        reg = outputs[..., self.num_classes: self.num_classes + 2]
        wh = outputs[..., -2:]
        heatmap = torch.sigmoid(heatmap)
        batch_size = heatmap.size()[0]
        heatmap = Decoder._nms(heatmap)
        scores, inds, classes, ys, xs = Decoder._top_k(scores=heatmap, k=self.K)
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

        bboxes /= self.feature_size
        bboxes = torch.clamp(bboxes, min=0, max=1)
        # (cx, cy, w, h) ----> (xmin, ymin, xmax, ymax)
        bboxes = xywh_to_xyxy_torch(bboxes)

        score_mask = scores >= self.score_threshold  # shape: (batch_size, self.K)

        bboxes = bboxes[score_mask]
        scores = scores[score_mask]
        classes = classes[score_mask]
        if self.use_nms:
            indices = diou_nms(boxes=bboxes, scores=scores, iou_threshold=self.score_threshold)
            bboxes, scores, classes = bboxes[indices], scores[indices], classes[indices]
        return bboxes, scores, classes

    @staticmethod
    def _nms(heatmap, pool_size=3):
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


def detect_one_image(cfg: Config, model, image_path, print_on, save_result, device):
    model.eval()
    # 处理单张图片
    image = read_image(image_path)
    h, w, c = image.shape
    image, _, _ = letter_box(image, cfg.arch.input_size[1:])
    image = TF.to_tensor(image).unsqueeze(0)
    image = image.to(device)

    decoder = Decoder(cfg,
                      input_image_size=cfg.arch.input_size[1:],
                      score_threshold=cfg.decode.score_threshold,
                      device=device)

    with torch.no_grad():
        preds = model(image)
        boxes, scores, classes = decoder(preds)
        # 将boxes坐标变换到原始图片上
        boxes = reverse_letter_box(h=h, w=w, input_size=cfg.arch.input_size[1:], boxes=boxes, xywh=False)

    show_detection_results(image_path=image_path,
                           dataset_name=cfg.dataset.dataset_name,
                           boxes=boxes,
                           scores=scores,
                           class_indices=classes,
                           print_on=print_on,
                           save_result=save_result,
                           save_dir=cfg.decode.test_results)
