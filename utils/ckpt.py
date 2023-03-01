import os

import torch


class CheckPoint:
    @staticmethod
    def check(path):
        """
        判断权重文件是否存在
        :param path:
        :return:
        """
        if path is None:
            return False
        return os.path.exists(path)

    @staticmethod
    def save(model, path, optimizer=None, scheduler=None):
        if optimizer is None and scheduler is None:
            # 仅保存模型的state_dict
            torch.save(model.state_dict(), path)
        else:
            obj = {"model": model.state_dict()}
            if optimizer is not None:
                obj["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                obj["scheduler"] = scheduler.state_dict()
            torch.save(obj, path)

    @staticmethod
    def load(path, device, model, pure=False, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location=device)
        if pure:
            # ckpt中仅保存了模型的state_dict
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt["model"])
            if optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
        del ckpt