# DeepLearning.pytorch
深度学习算法

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

## 样例
1. YOLOv7 on VOC2012

![sample 1](https://github.com/calmisential/Detection.pytorch/blob/main/performance/2010_006598%402023-03-24-12-36-57.jpg) 
![sample_2](https://github.com/calmisential/Detection.pytorch/blob/main/performance/2010_006639%402023-03-24-12-36-57.jpg)

## 训练好的模型
|                     模型名称                      | 输入图片尺寸（高，宽） |     VOC val      |      COCO val2017       |                                                                                     下载地址                                                                                     |  
|:---------------------------------------------:|:----:|:----------------:|:-----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  [YOLOv7](https://arxiv.org/abs/2207.02696)   | (640, 640)  | [mAP = 75.75%](https://github.com/calmisential/Detection.pytorch/blob/main/performance/yolov7_voc_val.txt) | [AP(0.5~0.95) = 48 %](https://github.com/calmisential/Detection.pytorch/blob/main/performance/yolov7_coco_val2017.txt) | [voc权重](https://github.com/calmisential/Detection.pytorch/releases/download/YOLOv7_weights-voc/YOLOv7_voc_final.pth) [coco权重](https://github.com/bubbliiiing/yolov7-pytorch) |
| [CenterNet](https://arxiv.org/abs/1904.07850) | (384, 384) | [mAP = 44.39%](https://github.com/calmisential/Detection.pytorch/blob/main/performance/centernet_voc_val.txt)|  |                         [voc权重](https://github.com/calmisential/Detection.pytorch/releases/download/CenterNet_weights_voc/CenterNet_voc_weights.pth)                         |

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
最后，修改`configs/dataset_cfg.py`文件中`VOC_CFG`和`COCO_CFG`中的`root`的值，分别为`${VOC_ROOT}/VOCdevkit/VOC2012/`和`${COCO_ROOT}/coco`

## 使用方法
### 训练
1. 修改`configs`文件夹下的模型配置文件中的参数
2. 修改`train.py`中的配置参数，将`MODE`改为0，然后运行`train.py`。

### 验证模型的性能
修改`evaluate.py`中的配置参数，验证模型在VOC或COCO数据集上的表现。

### 在图片（或视频）上测试
修改`detect.py`中的配置参数，然后运行`detect.py`。

## 参考
- https://github.com/bubbliiiing/ssd-pytorch
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/bubbliiiing/yolov7-pytorch
- https://github.com/ultralytics/ultralytics
