import torch


def warm_up_scheduler(optimizer, warmup_epochs, multi_step=True, milestones=None, gamma=None, last_epoch=-1):
    """
    warmup: 训练开始时从一个较小的学习率逐渐上升到初始学习率
    :param optimizer:  优化器
    :param warmup_epochs:  warmup的epoch数量
    :param multi_step: 是否使用MultiStepLR学习率衰减
    :param milestones:   MultiStepLR中的milestones参数
    :param gamma:   MultiStepLR中的gamma参数
    :param last_epoch:  -1表示从epoch-0开始
    :return:
    """
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            if not multi_step:
                return 1.0
            else:
                assert milestones is not None and gamma is not None, "milestones and gamma can't be None"
                factor = 1.0
                for m in milestones:
                    if (epoch + 1) > m:
                        factor *= gamma
                return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda, last_epoch=last_epoch)


def get_optimizer(optimizer_name, model, initial_lr):
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam([{"params": model.parameters(),
                                       'initial_lr': initial_lr}], lr=initial_lr)
    else:
        raise ValueError(f"{optimizer_name} is not supported")
    return optimizer

