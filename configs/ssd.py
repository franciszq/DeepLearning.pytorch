from utils.yaml_tools import load_yaml


class Config:
    def __init__(self):
        self.arch = self._Arch()
        self.dataset = self._Dataset()
        self.train = self._Train()
        self.optimizer = self._Optimizer()
        self.log = self._Log()
        self.decode = self._Decode()

    class _Arch:
        def __init__(self):
            # 目标类别数，与数据集有关，对于voc是20，对于coco是80
            self.num_classes = 20
            # 输入图片大小：(C, H, W)
            self.input_size = (3, 368, 368)
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
            self.anchor_size = [30, 60, 111, 162, 213, 264, 315]

    class _Dataset:
        # 数据集
        def __init__(self):
            # 数据集名称，"voc"或者"coco"
            self.dataset_name = "voc"
            # 数据集的配置文件
            cfgs = {
                "coco": load_yaml("configs/coco.yaml"),
                "voc": load_yaml("configs/voc.yaml")
            }
            self.dataset_config = cfgs[self.dataset_name]

    class _Train:
        # 训练参数
        def __init__(self):
            self.epoch = 100
            self.batch_size = 8
            # 是否使用预训练权重
            self.pretrained = False
            # 恢复训练时加载的checkpoint文件，""表示从epoch=0开始训练
            # 测试时也需要在这里指定checkpoint文件
            self.resume_training = ""
            # 模型保存间隔
            self.save_interval = 1
            # 每隔多少epoch在验证集上验证一次
            self.eval_interval = 1
            # 保存模型的文件夹
            self.save_path = "saves"
            # 是否启动tensorboard
            self.tensorboard_on = True
            # 是否使用混合精度训练
            self.mixed_precision = True
            # 多少个子进程用于数据加载
            self.num_workers = 0

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
            self.test_images_dir = "test"
            self.test_results = "result"