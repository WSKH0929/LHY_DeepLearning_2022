# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
# 定义训练循环
import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


# 定义学习率时间表
# 对于 Transformer 架构，学习率调度的设计与 CNN 不同。
# 以前的工作表明，学习率的预热对于训练具有变压器架构的模型很有用。
# 详见 https://zhuanlan.zhihu.com/p/410971793
def get_conine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1,
                                    ):
    """
    创建一个学习率随着优化器中设置的初始 lr 到 0 之间的余弦函数值而减小的计划，在预热期之后，它在 0 和优化器中设置的初始 lr 之间线性增加。
    :param optimizer: 为其安排学习率的优化器。
    :param num_warmup_steps:预热阶段的步数。
    :param num_training_steps:训练步骤的总数。
    :param num_cycles:余弦调度中的波数（默认值是从最大值减少到 0
    :param last_epoch:恢复训练时最后一个 epoch 的索引
    :return:学习了时间表
    """

    def lr_lambda(current_step):
        # 预热 Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 下降 Decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(train_loader, valid_loader, model, config, device):
    """ DNN training """

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hyper_paras'])

    scheduler = get_conine_schedule_with_warmup(optimizer, config['warmup_steps'], config['total_steps'])

    best_acc = -1  # 最佳模型的识别正确率
    train_init_loss, train_init_acc = valid(train_loader, model, config['device'])
    valid_init_loss, valid_init_acc = valid(valid_loader, model, config['device'])
    loss_record = {'train': [train_init_loss],
                   'val': [valid_init_loss]}  # for recording training loss
    acc_record = {'train': [train_init_acc],
                  'val': [valid_init_acc]}  # for recording training acc

    train_iterator = iter(train_loader)
    for step in range(config['total_steps']):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            # After each epoch, test your model on the validation (development) set.
            val_loss, val_acc = valid(valid_loader, model, device)
            loss_record['val'].append(val_loss)
            acc_record['val'].append(val_acc)
            if val_acc > best_acc:
                # Save model if your model improved
                best_acc = val_acc
                torch.save(model.state_dict(), config['model_save_dir'] + "model.pth")  # Save model to specified path
                print('[{:03d}/{:03d}] Val Acc: {:3.6f} loss: {:3.6f} -> best'.format(
                    step + 1, config['total_steps'], val_acc, val_loss
                ))
            else:
                print('[{:03d}/{:03d}] Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    step + 1, config['total_steps'], val_acc, val_loss
                ))

        # x : 32 * 10 * 40
        x, y = batch
        x, y = x.to(device), y.to(device)

        # 梯度下降五步走
        optimizer.zero_grad()
        pred = model(x)
        loss = model.cal_loss(pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        acc = (pred.argmax(-1) == y).float().mean().item()
        loss_record['train'].append(loss.item())
        acc_record['train'].append(acc)

    return loss_record


def valid(valid_loader, model, device):
    model.eval()  # set model to evaluation mode
    total_loss = 0
    val_acc = 0
    for x, y in valid_loader:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            loss = model.cal_loss(pred, y)  # compute loss
            # get the index of the class with the highest probability
            val_acc += (pred.argmax(dim=-1) == y).float().mean().item()

        total_loss += loss.item()  # accumulate loss
    total_loss = total_loss / len(valid_loader)  # compute averaged loss
    val_acc = val_acc / len(valid_loader)
    return total_loss, val_acc
