from configs.centernet import Config
from data.dataloader import PublicDataLoader
from data.transforms import TargetPadding


class CenterNetLoader(PublicDataLoader):
    def __init__(self, cfg: Config, dataset_name: str, batch_size, input_size):
        super().__init__(dataset_name, batch_size, input_size)
        self.train_transforms.insert(1, TargetPadding(max_num_boxes=cfg.train.max_num_boxes))