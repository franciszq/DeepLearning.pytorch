import numpy as np
import torch

from configs.yolo8_det_cfg import Config
from lib.models.yolov8.yolo_v8 import get_yolo8_n, get_yolo8_s, get_yolo8_m, get_yolo8_l, get_yolo8_x
from lib.utils.image_process import read_image_and_convert_to_tensor, read_image, yolo_correct_boxes
from lib.utils.ultralytics_ops import non_max_suppression
from lib.utils.visualize import show_detection_results


class YOLOv8:
    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device

        self.model_type = self.cfg.arch.model_type
        # 类别数目
        self.num_classes = self.cfg.dataset.num_classes
        # 输入图片的尺寸
        self.input_image_size = self.cfg.arch.input_size[1:]

        # 解码
        self.conf_threshold = self.cfg.decode.conf_threshold
        self.iou_threshold = self.cfg.decode.nms_threshold
        self.max_det = self.cfg.decode.max_det
        self.letterbox_image = self.cfg.decode.letterbox_image

    def build_model(self):
        """
        构建网络模型
        :return:
        """
        if self.model_type == "n":
            model = get_yolo8_n(nc=self.num_classes)
            model_name = "YOLOv8n"
        elif self.model_type == "s":
            model = get_yolo8_s(nc=self.num_classes)
            model_name = "YOLOv8s"
        elif self.model_type == "m":
            model = get_yolo8_m(nc=self.num_classes)
            model_name = "YOLOv8m"
        elif self.model_type == "l":
            model = get_yolo8_l(nc=self.num_classes)
            model_name = "YOLOv8l"
        elif self.model_type == "x":
            model = get_yolo8_x(nc=self.num_classes)
            model_name = "YOLOv8x"
        else:
            raise ValueError(f"model_type: {self.model_type} is not supported")
        return model, model_name

    def predict(self, model, image_path, print_on, save_result):
        """
        模型预测
        :param model:
        :param image_path:
        :param print_on:
        :param save_result:
        :return:
        """
        model.eval()
        # 处理单张图片
        image, h, w = read_image_and_convert_to_tensor(
            image_path, size=self.cfg.arch.input_size[1:], letterbox=self.letterbox_image)
        image = image.to(self.device)

        with torch.no_grad():
            # 图片输入到模型中，得到预测输出
            preds = model(image)
            # 解码
            results = self.decode_box(preds, h, w)

        if results[0].shape[0] == 0:
            print(f"No object detected")
            return read_image(image_path, mode='bgr')

        # 得到更详细的边界框信息、分数信息和类别信息
        boxes, scores, classes = results[0], results[1], results[2]

        # 将检测结果绘制在原始图片上
        return show_detection_results(image_path=image_path,
                                      dataset_name=self.cfg.dataset.dataset_name,
                                      boxes=boxes,
                                      scores=scores,
                                      class_indices=classes,
                                      print_on=print_on,
                                      save_result=save_result,
                                      save_dir=self.cfg.decode.test_results)

    def decode_box(self, preds, image_h, image_w, conf_threshold=None):
        """
        解码预测结果
        :param preds: 预测结果
        :param image_h: 图片高度
        :param image_w: 图片宽度
        :param conf_threshold: 置信度
        :return: 解码后的预测结果
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        preds = non_max_suppression(preds,
                                    conf_threshold,
                                    self.iou_threshold,
                                    agnostic=False,
                                    max_det=self.max_det,
                                    classes=None)

        assert len(preds) == 1, "仅支持单张图片的预测"
        pred = preds[0].cpu().numpy()
        bbox, conf, cls = pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int)
        # 坐标归一化到0~1范围内
        bbox[:, ::2] /= self.input_image_size[1]
        bbox[:, 1::2] /= self.input_image_size[0]

        # 计算相对于原始输入图片的边界框坐标
        box_xy, box_wh = (bbox[:, 0:2] + bbox[:, 2:4]
                          ) / 2, bbox[:, 2:4] - bbox[:, 0:2]
        bbox[:, :4] = yolo_correct_boxes(box_xy, box_wh,
                                         self.input_image_size, [image_h, image_w],
                                         self.letterbox_image)
        return bbox, conf, cls
