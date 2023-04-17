import copy
import os
import torch

from registry import model_registry


class CheckPointModel:
    def __init__(self, cfg_path: str, checkpoint_path: str, device):
        """
        :param cfg_path: 配置文件路径
        :param checkpoint_path: checkpoint文件路径
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        cfg = os.path.basename(cfg_path)
        try:
            self.model_cfg, self.model_class, self.trainer = model_registry[cfg]
        except KeyError:
            raise ValueError(f"找不到配置文件：{cfg_path}.")

        self.model, _ = self.model_class(self.model_cfg, device).build_model()
        self._to_weights()

    def _to_weights(self):
        """
        将checkpoint文件转换为单纯的模型权重文件
        :return:
        """
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        print(f"Successfully loaded checkpoint from {self.checkpoint_path}")
        del ckpt

    def save_as_weights(self, filepath):
        """
        保存为权重文件
        :param filepath:
        :return:
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Successfully saved weights to {filepath}")


if __name__ == '__main__':
    ckpt_model = CheckPointModel(cfg_path="configs/centernet_cfg.py",
                                 checkpoint_path="saves/CenterNet_coco_epoch-15.pth",
                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_model.save_as_weights("saves/CenterNet_coco_weights.pth")
