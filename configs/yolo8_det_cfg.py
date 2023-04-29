from configs.dataset_cfg import VOC_CFG, COCO_CFG


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
            # YOLOv8的模型种类：n, s, m, l, x
            self.model_type = "m"
            # 输入图片大小：(C, H, W)
            self.input_size = (3, 640, 640)

    class _Dataset:
        # 数据集
        def __init__(self):
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = COCO_CFG["num_classes"]
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = COCO_CFG["name"]

    class _Train:
        # 训练参数
        def __init__(self):
            # 恢复训练时加载的checkpoint文件，""表示从epoch=0开始训练
            # 测试时也需要在这里指定checkpoint文件
            self.resume_training = ""
            # 恢复训练时的上一次epoch是多少，-1表示从epoch=0开始训练
            self.last_epoch = -1

            self.epoch = 100
            self.batch_size = 8
            # 初始学习率
            self.initial_lr = 1e-3
            # warm up轮数
            self.warmup_epochs = 0
            self.milestones = []
            self.gamma = 0.1

            # 是否使用预训练权重
            self.pretrained = False
            self.pretrained_weights = ""
            # 模型保存间隔
            self.save_interval = 1
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
            self.box = 7.5
            self.cls = 0.5
            self.dfl = 1.5

    class _Optimizer:
        # 优化器
        def __init__(self):
            self.name = "Adam"

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
            self.conf_threshold = 0.25
            self.nms_threshold = 0.7
            # 单张图片上检测框的最大数量
            self.max_det = 300
