# Detection.pytorch
部分目标检测算法的pytorch实现

[English](https://github.com/calmisential/Detection.pytorch/blob/main/README_EN.md)
## 安装
确保安装以下版本的开发环境：
- cuda: 11.6
- torch: 1.13.1
- torchvision: 0.14.1

在终端运行
```commandline
git clone https://github.com/calmisential/Detection.pytorch.git
cd Detection.pytorch
pip install -r requirements.txt
```

## 准备数据集
- 对于COCO数据集，从[这里](https://cocodataset.org/#download)下载2017 Train/Val images和2017 Train/Val annotations，
解压之后放在`${COCO_ROOT}`文件夹中，目录结构如下：
```
|-- coco
`-----|-- annotations
      |   |-- captions_train2017.json
      |   |-- captions_val2017.json
      |   |-- instances_train2017.json
      |   |-- instances_val2017.json
      |   |-- person_keypoints_train2017.json
      |   `-- person_keypoints_val2017.json
      |
       `-- images
            |-- train2017
            |   |-- ... 
            `-- val2017
                |-- ... 
```
- 对于VOC2012数据集，从[这里](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit)下载，
解压之后放在`${VOC_ROOT}`文件夹中，目录结构如下：
```
|--VOCdevkit
      |---VOC2012
           |
            `-- |-- Annotations
                |-- ImageSets
                |-- JPEGImages
                |-- SegmentationClass
                `-- SegmentationObject
```
最后，修改`coco.yaml`和`voc.yaml`文件中参数`root`的值，分别为`${COCO_ROOT}/coco`和`${VOC_ROOT}/VOCdevkit/VOC2012/`

## 使用方法
### 训练
修改`train.py`中的配置参数，将`MODE`改为0，然后运行`train.py`。

### 验证模型的性能
修改`train.py`中的配置参数，将`MODE`改为1，然后运行`train.py`。

### 在图片上测试
修改`detect.py`中的配置参数，然后运行`detect.py`。

## 参考
- https://github.com/bubbliiiing/ssd-pytorch
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/bubbliiiing/yolov7-pytorch