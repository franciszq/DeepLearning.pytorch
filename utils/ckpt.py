import torch


class CheckPoint:
    @staticmethod
    def save(model, optimizer, scheduler, epoch, path):
        if scheduler is None:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scheduler": "None"
            }, path)
        else:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scheduler": scheduler.state_dict()
            }, path)

    @staticmethod
    def load(path, device, model, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        epoch = ckpt["epoch"]
        del ckpt
        return model, optimizer, scheduler, epoch