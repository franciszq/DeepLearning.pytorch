import torch


def warm_up_scheduler(optimizer, warmup_epochs, milestones=None, gamma=0.1, last_epoch=-1):
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            if milestones is None:
                return 1.0
            else:
                factor = 1.0
                for m in milestones:
                    if (epoch + 1) > m:
                        factor *= gamma
                return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda, last_epoch=last_epoch)
