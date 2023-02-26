import torch

from utils.anchor import get_yolo7_anchors
from utils.image_process import read_image_and_convert_to_tensor
from utils.nms import yolo7_nms
from configs.yolo7 import Config
from utils.visualize import show_detection_results


def detect_one_image(cfg: Config, model, image_path, print_on, save_result, device):
    model.eval()
    # 处理单张图片
    image = read_image_and_convert_to_tensor(image_path, size=cfg.arch.input_size[1:], letterbox=True)
    image = image.to(device)
    _, _, h, w = image.size()
    # 生成anchors
    anchors = get_yolo7_anchors(cfg)

    decoder = Decoder(anchors=anchors.copy(),
                      num_classes=cfg.arch.num_classes,
                      input_shape=cfg.arch.input_size[1:],
                      anchors_mask=cfg.arch.anchors_mask,
                      ori_image_shape=[h, w],
                      letterbox_image=True,
                      conf_threshold=cfg.decode.conf_threshold,
                      nms_threshold=cfg.decode.nms_threshold,
                      device=device)

    with torch.no_grad():
        preds = model(image)
        results = decoder(preds)

    if results[0] is None:
        print(f"No object detected")
        return

    boxes = torch.from_numpy(results[0][:, :4])
    scores = torch.from_numpy(results[0][:, 4] * results[0][:, 5])
    classes = torch.from_numpy(results[0][:, 6]).to(torch.int32)

    show_detection_results(image_path=image_path,
                           dataset_name=cfg.dataset.dataset_name,
                           boxes=boxes,
                           scores=scores,
                           class_indices=classes,
                           print_on=print_on,
                           save_result=save_result,
                           save_dir=cfg.decode.test_results)


class Decoder:
    def __init__(self,
                 anchors,
                 num_classes,
                 input_shape,
                 anchors_mask,
                 ori_image_shape,
                 letterbox_image,
                 conf_threshold,
                 nms_threshold,
                 device):
        """
        :param anchors:
        :param num_classes:
        :param input_shape:
        :param anchors_mask:
        :param ori_image_shape:
        :param letterbox_image: 是否使用了保持宽高比的resize方式
        :param conf_threshold:
        :param nms_threshold:
        :param device:
        """
        self.device = device
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.image_shape = ori_image_shape
        self.letterbox_image = letterbox_image
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def set_h_w(self, h, w):
        self.image_shape = [h, w]

    def __call__(self, predictions):
        outputs = self.decode_box(predictions)
        results = yolo7_nms(prediction=outputs,
                            num_classes=self.num_classes,
                            input_shape=self.input_shape,
                            image_shape=self.image_shape,
                            letterbox_image=self.letterbox_image,
                            device=self.device,
                            conf_thres=self.conf_threshold,
                            nms_thres=self.nms_threshold)
        return results

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size = 1
            #   batch_size, 3 * (4 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            bs, _, input_height, input_width = input.size()
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]
            # -----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            prediction = torch.reshape(input, shape=(bs, len(self.anchors_mask[i]), self.bbox_attrs, input_height,
                                                     input_width)).permute(0, 1, 3, 4, 2)
            # -----------------------------------------------#
            #   先验框的中心位置的调整参数
            # -----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            # -----------------------------------------------#
            #   先验框的宽高调整参数
            # -----------------------------------------------#
            w = torch.sigmoid(prediction[..., 2])
            h = torch.sigmoid(prediction[..., 3])
            # -----------------------------------------------#
            #   获得置信度，是否有物体
            # -----------------------------------------------#
            conf = torch.sigmoid(prediction[..., 4])
            # -----------------------------------------------#
            #   种类置信度
            # -----------------------------------------------#
            pred_cls = torch.sigmoid(prediction[..., 5:])
            # ----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            grid_x = torch.linspace(start=0, end=input_width - 1, steps=input_width).repeat(input_height, 1).repeat(
                bs * len(self.anchors_mask[i]), 1, 1).view(x.shape).to(torch.float32).to(self.device)
            grid_y = torch.linspace(start=0, end=input_height - 1, steps=input_height).repeat(input_width,
                                                                                              1).t().repeat(
                bs * len(self.anchors_mask[i]), 1, 1).view(y.shape).to(torch.float32).to(self.device)

            # ----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,20,20
            # ----------------------------------------------------------#
            anchor_w = torch.tensor(scaled_anchors, dtype=torch.float32,
                                    device=self.device).index_select(1, torch.tensor([0],
                                                                                     dtype=torch.int64,
                                                                                     device=self.device))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, input_height * input_width).view(w.shape)

            anchor_h = torch.tensor(scaled_anchors, dtype=torch.float32,
                                    device=self.device).index_select(1, torch.tensor([1],
                                                                                     dtype=torch.int64,
                                                                                     device=self.device))
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, input_height * input_width).view(h.shape)
            # ----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => 负责一定范围的目标的预测
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # ----------------------------------------------------------#
            pred_boxes = torch.zeros_like(prediction[..., :4])
            pred_boxes[..., 0] = x.detach() * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.detach() * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.detach() * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.detach() * 2) ** 2 * anchor_h
            # ----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            # ----------------------------------------------------------#
            _scale = torch.tensor([input_width, input_height, input_width, input_height],
                                  dtype=torch.float32,
                                  device=self.device)
            output = torch.cat((pred_boxes.reshape(bs, -1, 4) / _scale,
                                conf.reshape(bs, -1, 1), pred_cls.reshape(bs, -1, self.num_classes)), -1)
            outputs.append(output.detach())
        return torch.cat(outputs, dim=1)
