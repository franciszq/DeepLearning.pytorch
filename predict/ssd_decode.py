import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms, nms
import torchvision.transforms.functional as TF

from utils.anchor import generate_ssd_anchor, generate_ssd_anchor_v2
from utils.bboxes import xyxy_to_xywh
from utils.image_process import reverse_letter_box, read_image, letter_box, yolo_correct_boxes
from configs.ssd import Config
from utils.visualize import show_detection_results


class Decoder:
    def __init__(self,
                 anchors,
                 input_image_size,
                 num_max_output_boxes,
                 num_classes,
                 variance,
                 conf_threshold,
                 nms_threshold,
                 device):
        """
        :param anchors:  先验框，numpy.ndarray, shape: (8732, 4)
        :param input_image_size:  SSD网络的输入图片大小
        :param num_max_output_boxes: 最大输出检测框数量
        :param num_classes:  检测的目标类别数
        :param variance:
        :param conf_threshold:
        :param nms_threshold:
        :param device: 设备
        """
        self.device = device
        self.priors_xywh = torch.from_numpy(xyxy_to_xywh(anchors)).to(device)
        self.priors_xywh = torch.unsqueeze(self.priors_xywh, dim=0)
        self.top_k = num_max_output_boxes
        self.num_classes = num_classes + 1
        self.scale_xy = variance[0]
        self.scale_wh = variance[1]

        self.conf_thresh = conf_threshold
        self.nms_thresh = nms_threshold

        self.input_image_size = input_image_size

    def _decode(self, bboxes_in, scores_in):
        """
        :param bboxes_in: torch.Tensor, shape: (batch_size, num_priors, 4)
        :param scores_in: torch.Tensor, shape: (batch_size, num_priors, self.num_classes)
        :return:
        """
        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]
        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.priors_xywh[:, :, 2:] + self.priors_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = torch.exp(bboxes_in[:, :, 2:]) * self.priors_xywh[:, :, 2:]

        xmin = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
        xmax = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
        ymin = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
        ymax = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]
        bboxes_in[:, :, 0] = xmin
        bboxes_in[:, :, 1] = ymin
        bboxes_in[:, :, 2] = xmax
        bboxes_in[:, :, 3] = ymax
        return bboxes_in, F.softmax(scores_in, dim=-1)

    def __call__(self, outputs):
        loc_data, conf_data = outputs

        bboxes, probs = self._decode(loc_data, conf_data)

        batch_size = bboxes.size(0)
        assert batch_size == 1, "仅支持单张图片作为预测输入。"
        bbox = torch.squeeze(bboxes, dim=0)  # (num_priors, 4)
        prob = torch.squeeze(probs, dim=0)  # (num_priors, self.num_classes)
        num_priors = self.priors_xywh.size(0)

        # 限制在[0, 1]范围内
        decoded_boxes = torch.clamp(bbox, min=0, max=1)
        # (num_priors, 4) - > (num_priors, 21, 4)
        decoded_boxes = decoded_boxes.repeat(1, self.num_classes).reshape(prob.size()[0], -1, 4)

        # 为每一个类别创建标签信息(0~20)
        labels = torch.arange(self.num_classes, dtype=torch.int32, device=self.device)
        labels = labels.view(1, -1).expand_as(prob)  # shape: (num_priors, num_classes)

        # 移除背景类别的信息
        boxes = decoded_boxes[:, 1:, :]  # (num_priors, 20, 4)
        scores = prob[:, 1:]  # (num_priors, 20)
        labels = labels[:, 1:]  # (num_priors, 20)

        # 对于每一个box，它都有可能属于这(num_classes-1)个类别之一
        boxes_all = boxes.reshape(-1, 4)  # (num_priors*20, 4)
        scores_all = scores.reshape(-1)  # (num_priors*20)
        labels_all = labels.reshape(-1)  # (num_priors*20)

        # 移除低概率目标
        inds = torch.nonzero(scores_all > self.conf_thresh).squeeze(1)
        boxes_all, scores_all, labels_all = boxes_all[inds, :], scores_all[inds], labels_all[inds]

        # 移除面积很小的边界框
        w, h = boxes_all[:, 2] - boxes_all[:, 0], boxes_all[:, 3] - boxes_all[:, 1]
        keep = (w >= 1 / self.input_image_size[1]) & (h >= 1 / self.input_image_size[0])
        keep = keep.nonzero().squeeze(1)
        boxes_all, scores_all, labels_all = boxes_all[keep], scores_all[keep], labels_all[keep]

        boxes_all, scores_all, labels_all = boxes_all.to(torch.float32), scores_all.to(
            torch.float32), labels_all.to(torch.int32)

        # nms
        keep = batched_nms(boxes_all, scores_all, labels_all, iou_threshold=self.nms_thresh)
        keep = keep[:self.top_k]
        boxes_out = boxes_all[keep]
        scores_out = scores_all[keep]
        labels_out = labels_all[keep]

        # # 将boxes坐标变换到原始图片上
        # boxes = reverse_letter_box(h=self.original_image_size[0], w=self.original_image_size[1],
        #                            input_size=self.input_image_size, boxes=boxes_out, xywh=False)

        return boxes_out, scores_out, labels_out - 1


class DecoderV2:
    def __init__(self,
                 anchors,
                 ori_image_shape,
                 input_image_size,
                 num_max_output_boxes,
                 num_classes,
                 variance,
                 conf_threshold,
                 nms_threshold,
                 device):
        """
        :param anchors:  先验框，numpy.ndarray, shape: (8732, 4)
        :param ori_image_shape: [h, w] 待检测图片大小
        :param input_image_size:  SSD网络的输入图片大小
        :param num_max_output_boxes: 最大输出检测框数量
        :param num_classes:  检测的目标类别数
        :param variance:
        :param conf_threshold:
        :param nms_threshold:
        :param device: 设备
        """
        self.device = device
        self.anchors = torch.from_numpy(anchors).to(torch.float32).to(device)
        # self.priors_xywh = torch.unsqueeze(self.priors_xywh, dim=0)
        # self.top_k = num_max_output_boxes
        self.num_classes = num_classes + 1
        # self.scale_xy = variance[0]
        # self.scale_wh = variance[1]
        self.variance = variance

        self.conf_thresh = conf_threshold
        self.nms_thresh = nms_threshold

        self.input_image_size = input_image_size
        self.image_shape = ori_image_shape

    def set_h_w(self, h, w):
        self.image_shape = [h, w]

    def decode_boxes(self, mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width *= anchor_width
        decode_bbox_height = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height *= anchor_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                 decode_bbox_ymin[:, None],
                                 decode_bbox_xmax[:, None],
                                 decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def __call__(self, predictions):
        """
        :param predictions:  模型的预测输出
        :return:
        """
        # ---------------------------------------------------#
        #   :4是回归预测结果
        # ---------------------------------------------------#
        mbox_loc = predictions[0]
        # ---------------------------------------------------#
        #   获得种类的置信度
        # ---------------------------------------------------#
        mbox_conf = nn.Softmax(-1)(predictions[1])

        results = []
        # ----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        # ----------------------------------------------------------------------------------------------------------------#
        for i in range(len(mbox_loc)):
            results.append([])
            # --------------------------------#
            #   利用回归结果对先验框进行解码
            # --------------------------------#
            decode_bbox = self.decode_boxes(mbox_loc[i], self.anchors, self.variance)

            for c in range(1, self.num_classes):
                # --------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                # --------------------------------#
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > self.conf_thresh
                if len(c_confs[c_confs_m]) > 0:
                    # -----------------------------------------#
                    #   取出得分高于confidence的框
                    # -----------------------------------------#
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        self.nms_thresh
                    )
                    # -----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    # -----------------------------------------#
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones(
                        (len(keep), 1))
                    # -----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    # -----------------------------------------#
                    c_pred = torch.cat((good_boxes, labels, confs), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:,
                                                                                                        0:2]
                results[-1][:, :4] = yolo_correct_boxes(box_xy, box_wh, self.input_image_size, self.image_shape,
                                                        True)

        return results


def detect_one_image(cfg: Config, model, image_path, print_on, save_result, device):
    model.eval()
    # 处理单张图片
    image = read_image(image_path)
    h, w, c = image.shape
    image, _, _ = letter_box(image, cfg.arch.input_size[1:])
    image = TF.to_tensor(image).unsqueeze(0)
    image = image.to(device)

    # 生成anchor
    anchors = generate_ssd_anchor_v2(input_image_shape=cfg.arch.input_size[1:],
                                     anchor_sizes=cfg.arch.anchor_size,
                                     feature_shapes=cfg.arch.feature_shapes,
                                     aspect_ratios=cfg.arch.aspect_ratios)  # (8732, 4)

    # decoder = Decoder(anchors=anchors.copy(),
    #                   input_image_size=cfg.arch.input_size[1:],
    #                   num_max_output_boxes=cfg.decode.num_max_output_boxes,
    #                   num_classes=cfg.arch.num_classes,
    #                   variance=cfg.loss.variance,
    #                   conf_threshold=cfg.decode.confidence_threshold,
    #                   nms_threshold=cfg.decode.nms_threshold,
    #                   device=device)

    decoder = DecoderV2(anchors=anchors.copy(),
                        ori_image_shape=[h, w],
                        input_image_size=cfg.arch.input_size[1:],
                        num_max_output_boxes=cfg.decode.num_max_output_boxes,
                        num_classes=cfg.arch.num_classes,
                        variance=cfg.loss.variance,
                        conf_threshold=cfg.decode.confidence_threshold,
                        nms_threshold=cfg.decode.nms_threshold,
                        device=device)

    # with torch.no_grad():
    #     preds = model(image)
    #     boxes, scores, classes = decoder(preds)
    #     # 将boxes坐标变换到原始图片上
    #     boxes = reverse_letter_box(h=h, w=w, input_size=cfg.arch.input_size[1:], boxes=boxes, xywh=False)

    with torch.no_grad():
        preds = model(image)
        results = decoder(preds)

    if len(results[0]) == 0:
        print(f"No object detected")
        return

    boxes = torch.from_numpy(results[0][:, :4])
    scores = torch.from_numpy(results[0][:, 5])
    classes = torch.from_numpy(results[0][:, 4]).to(torch.int32)

    show_detection_results(image_path=image_path,
                           dataset_name=cfg.dataset.dataset_name,
                           boxes=boxes,
                           scores=scores,
                           class_indices=classes,
                           print_on=print_on,
                           save_result=save_result,
                           save_dir=cfg.decode.test_results)
