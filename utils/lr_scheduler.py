import torch


def warm_up_scheduler(optimizer, warmup_epochs, last_epoch=-1):
    def warmup_lambda(epoch):
        if warmup_epochs == 0:
            return 1.0
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda, last_epoch=last_epoch)