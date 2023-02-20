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


def get_optimizer(optimizer_name, model, initial_lr):
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam([{"params": model.parameters(),
                                       'initial_lr': initial_lr}], lr=initial_lr)
    else:
        raise ValueError(f"{optimizer_name} is not supported")
    return optimizer


def get_lr_scheduler(scheduler_name, optimizer, last_epoch, **kwargs):
    if scheduler_name.lower() == "multi_step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=kwargs["milestones"],
                                                            gamma=kwargs["gamma"],
                                                            last_epoch=last_epoch)
    elif scheduler_name.lower() == "None":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0, last_epoch=last_epoch)
    else:
        raise ValueError(f"{scheduler_name} is not supported")
    return lr_scheduler
