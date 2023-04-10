from .dataset_cfg import COCO_CFG, VOC_CFG


class Config:
    def __init__(self):
        self.arch = self._Arch()
        self.dataset = self._Dataset()
        self.train = self._Train()
        self.loss = self._Loss()
        self.optimizer = self._Optimizer()
        self.log = self._Log()
        self.decode = self._Decode()

    class _Arch:
        def __init__(self):
            self.backbone = "vgg16"   # 'vgg16' or 'mobilenetv1'
            # 输入图片大小：(C, H, W)
            self.input_size = (3, 300, 300)
            # 先验框的宽高比
            self.aspect_ratios = [[1, 2, 1.0 / 2],
                                  [1, 2, 1.0 / 2, 3, 1.0 / 3],
                                  [1, 2, 1.0 / 2, 3, 1.0 / 3],
                                  [1, 2, 1.0 / 2, 3, 1.0 / 3],
                                  [1, 2, 1.0 / 2],
                                  [1, 2, 1.0 / 2]]
            # ssd结构中6个特征层的输出通道数
            self.feature_channels = [512, 1024, 512, 256, 256, 256]
            self.feature_shapes = [38, 19, 10, 5, 3, 1]
            # 先验框的宽和高
            self.anchor_sizes = [30, 60, 111, 162, 213, 264, 315]

    class _Dataset:
        # 数据集
        def __init__(self):
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = VOC_CFG["num_classes"]
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = VOC_CFG["name"]

    class _Train:
        # 训练参数
        def __init__(self):
            # 恢复训练时加载的checkpoint文件，""表示从epoch=0开始训练
            self.resume_training = ""
            # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
            self.last_epoch = -1

            self.epoch = 100
            self.batch_size = 16
            # 初始学习率
            self.initial_lr = 1e-3
            # warm up轮数
            self.warmup_epochs = 0
            self.milestones = []
            self.gamma = 0.1

            # 是否使用预训练权重
            self.pretrained = False
            # 预训练模型的权重路径
            self.pretrained_weights = ""
            # 模型保存间隔
            self.save_interval = 5
            # 每隔多少epoch在验证集上验证一次
            self.eval_interval = 0
            # 保存模型的文件夹
            self.save_path = "saves"
            # 是否启动tensorboard
            self.tensorboard_on = True
            # 是否使用混合精度训练
            self.mixed_precision = True
            # 多少个子进程用于数据加载
            self.num_workers = 0

    class _Loss:
        # 损失函数
        def __init__(self):
            self.alpha = 0.25
            self.gamma = 2.0
            self.overlap_threshold = 0.5
            self.neg_pos = 3
            self.variance = [0.1, 0.2]

    class _Optimizer:
        # 优化器
        def __init__(self):
            self.name = "Adam"
            self.scheduler_name = "multi_step"

    class _Log:
        # 训练日志
        def __init__(self):
            # 日志文件保存文件夹
            self.root = "out"
            # 日志文件输出间隔
            self.print_interval = 50

    class _Decode:
        def __init__(self):
            self.test_results = "result"
            self.letterbox_image = True
            self.nms_threshold = 0.5
            self.confidence_threshold = 0.7
