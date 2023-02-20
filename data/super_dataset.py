import os

import torch
import cv2
import numpy as np
import xml.dom.minidom as xdom
from torch.utils.data import Dataset

from utils.image_process import read_image
from utils.yaml_tools import load_yaml


class SuperDataset(Dataset):
    def __init__(self, dataset_name):
        """

        :param dataset_name: 数据集名称, 'voc' or 'coco'
        """
        super().__init__()
        assert dataset_name in ['voc', 'coco'], f"Unsupported dataset: {dataset_name}"
        self.dataset_name = dataset_name
        if dataset_name == 'voc':
            self.root, self.class_names, self.images, self.class2index = self._parse_voc()
        else:
            # TODO COCO数据集解析
            pass



    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.dataset_name == 'voc':
            # 使用opencv读取图片
            image_path = self.images[item]
            image = read_image(image_path)

            # 所有xml文件路径的列表
            xml_paths = [os.path.join(self.root, "Annotations", f"{e}.xml") for e in self.images]
            # 解析xml
            target = self._parse_xml(xml_paths[item])
            target = np.array(target, dtype=np.float32)
            target = np.reshape(target, (-1, 5))  # shape: (N, 5)  N是这张图片包含的目标数



        else:
            pass





    def _get_voc_root_and_classes(self, voc_yaml):
        cfg_dict = load_yaml(voc_yaml)
        return cfg_dict["root"], cfg_dict["classes"]

    def _parse_voc(self):
        # VOC数据集的根目录和类别名
        voc_root, voc_class_names = self._get_voc_root_and_classes("configs/voc.yaml")
        images_root = os.path.join(voc_root, "JPEGImages")
        # 加载训练集
        train_txt = os.path.join(voc_root, "ImageSets", "Main", "train.txt")
        with open(train_txt, mode="r", encoding="utf-8") as f:
            image_names = f.read().strip().split('\n')
        # 所有图片路径的列表
        image_paths = [os.path.join(images_root, f"{e}.jpg") for e in image_names]
        # voc类别名的列表
        class2index = dict((v, k) for k, v in enumerate(voc_class_names))

        return voc_root, voc_class_names, image_paths, class2index


    def _parse_xml(self, xml):
        box_class_list = []
        DOMTree = xdom.parse(xml)
        annotation = DOMTree.documentElement
        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bbox = o.getElementsByTagName("bndbox")[0]
            xmin = bbox.getElementsByTagName("xmin")[0].childNodes[0].data
            ymin = bbox.getElementsByTagName("ymin")[0].childNodes[0].data
            xmax = bbox.getElementsByTagName("xmax")[0].childNodes[0].data
            ymax = bbox.getElementsByTagName("ymax")[0].childNodes[0].data
            o_list.append(float(xmin))
            o_list.append(float(ymin))
            o_list.append(float(xmax))
            o_list.append(float(ymax))
            o_list.append(self.class2index[obj_name])
            box_class_list.append(o_list)
        # [[xmin, ymin, xmax, ymax, class_index], ...]
        return box_class_list

    def _parse_coco(self):
        pass