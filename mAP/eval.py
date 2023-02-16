import os

import torch
from tqdm import tqdm

from data.voc import get_voc_root_and_classes
from utils.image_process import read_image, letter_box, reverse_letter_box
import torchvision.transforms.functional as TF


def evaluate_pipeline(model,
                      decoder,
                      input_image_size,
                      result_root,
                      dataset='voc',
                      subset='val',
                      device=None):
    """
    验证模型性能的完整管道
    :param model: 网络模型
    :param decoder: 此模型对应的解码器
    :param input_image_size:  [h, w], 模型输入图片大小
    :param result_root:  检测结果（txt文件）保存的根目录
    :param dataset:
    :param subset: 子集，'val' or 'test'
    :param device: 设备
    :return:
    """
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    voc_root, voc_class_names = get_voc_root_and_classes("configs/voc.yaml")
    if subset == 'val':
        image_ids = open(os.path.join(voc_root, "ImageSets", "Main", "val.txt"), mode='r').read().strip().split(
            '\n')
    elif subset == 'test':
        image_ids = open(os.path.join(voc_root, "ImageSets", "Main", "test.txt"), mode='r').read().strip().split(
            '\n')
    else:
        raise ValueError(f"sub_set must be one of 'test' and 'val', but got {subset}")
    with tqdm(image_ids, desc=f"Evaluate on voc-{subset}") as pbar:
        for image_id in pbar:
            # 图片预处理
            image_path = os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg")
            image = read_image(image_path)
            h, w, c = image.shape
            image, _, _ = letter_box(image, input_image_size)
            image = TF.to_tensor(image).unsqueeze(0)
            image = image.to(device)

            # 得到检测结果
            with torch.no_grad():
                preds = model(image)
                boxes, scores, classes = decoder(preds)
                # 将boxes坐标变换到原始图片上
                boxes = reverse_letter_box(h=h, w=w, input_size=input_image_size, boxes=boxes, xywh=False)

            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            class_indices = classes.cpu().numpy().tolist()

            # 将检测结果写入txt文件中
            with open(file=os.path.join(result_root, f"{image_id}.txt"), mode='w', encoding='utf-8') as f:
                for i, c in enumerate(class_indices):
                    predicted_class = voc_class_names[int(c)]
                    score = str(scores[i])

                    top = boxes[i, 1]  # ymin
                    left = boxes[i, 0]  # xmin
                    bottom = boxes[i, 3]  # ymax
                    right = boxes[i, 2]  # xmax

                    f.write(f"{predicted_class} {score[:6]} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")
