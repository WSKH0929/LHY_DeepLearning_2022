# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15

import numpy as np
import torch
import random
from torch.utils.data import random_split
import matplotlib.pyplot as plt


def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility
    固定时间种子。由于cuDNN会自动从几种算法中寻找最适合当前配置的算法，为了使选择的算法固定，所以固定时间种子
    :param seed: 时间种子
    :return: None
    """
    torch.backends.cudnn.deterministic = True  # 解决算法本身的不确定性，设置为True 保证每次结果是一致的
    torch.backends.cudnn.benchmark = False  # 解决了算法选择的不确定性，方便复现，提升训练速度
    np.random.seed(seed)  # 按顺序产生固定的数组，如果使用相同的seed，则生成的随机数相同， 注意每次生成都要调用一次
    torch.manual_seed(seed)  # 手动设置torch的随机种子，使每次运行的随机数都一致
    random.seed(seed)
    if torch.cuda.is_available():
        # 为GPU设置唯一的时间种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """
    Split provided training data into training set and validation set
    将数据集随机分割为 训练集 和 验证集
    :param data_set: 源数据
    :param valid_ratio: 验证集比例
    :param seed: 时间种子
    :return: 训练集 验证集
    """
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def select_feature(train_data, valid_data, test_data, select_all=True):
    """
    Select useful features to perform regression
    :param train_data: 训练集
    :param valid_data: 验证集
    :param test_data: 测试集
    :param select_all: 选择前五行还是全部数据
    :return:
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    row_x_train, row_x_valid, row_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    if select_all:
        feat_idx = list(range(row_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # 选择前五行数据
    return row_x_train[:, feat_idx], row_x_valid[:, feat_idx], row_x_test[:, feat_idx], y_train, y_valid


def plot_learning_curve(loss_record, title='', save_dir=None):
    """
    Plot learning curve of your Model (train & dev loss)
    """
    x_1 = range(len(loss_record['train']))
    if len(loss_record['val']) == 1:
        x_2 = [0]
    else:
        stride = len(loss_record['train']) // (len(loss_record['val']) - 1)
        if len(loss_record['val']) == 2:
            stride -= 1
        x_2 = x_1[::stride]
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['val'], c='tab:cyan', label='val')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.grid(False)
    if save_dir is not None:
        plt.savefig(save_dir + "learning_curve.svg")
    plt.show()

def plot_acc_curve(acc_record, title='', save_dir=None):
    """
    Plot learning curve of your Model (train & dev loss)
    """
    x_1 = range(len(acc_record['train']))
    if len(acc_record['val']) == 1:
        x_2 = [0]
    else:
        stride = len(acc_record['train']) // (len(acc_record['val']) - 1)
        if len(acc_record['val']) == 2:
            stride -= 1
        x_2 = x_1[::stride]
    plt.plot(x_1, acc_record['train'], c='tab:red', label='train')
    plt.plot(x_2, acc_record['val'], c='tab:cyan', label='val')
    plt.xlabel('Training steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve of {}'.format(title))
    plt.legend()
    plt.grid(False)
    if save_dir is not None:
        plt.savefig(save_dir + "accuracy_curve.svg")
    plt.show()

def plot_GAN_learning_curve(loss_record, save_dir=None):
    x = range(len(loss_record['G']))
    # Generator
    plt.figure()
    plt.plot(x, loss_record['G'], c='tab:red')
    plt.xlabel('Generator Updates')
    plt.ylabel('Loss')
    plt.title('Learning curve of Generator')
    plt.grid(False)
    if save_dir is not None:
        plt.savefig(save_dir + "generator_learning_curve.svg")
    plt.show()

    # Discriminator
    plt.figure()
    plt.plot(x, loss_record['D'], c='tab:red')
    plt.xlabel('Discriminator Updates')
    plt.ylabel('Loss')
    plt.title('Learning curve of Discriminator')
    plt.grid(False)
    if save_dir is not None:
        plt.savefig(save_dir + "discriminator_learning_curve.svg")
    plt.show()


def plot_valid_pred(valid_loader, model, device, preds=None, targets=None, save_dir=None):
    """ Plot prediction of your Model """
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    min_value, max_value = None, None
    for i in range(preds.shape[0]):
        if min_value is None or min_value > preds[i]:
            min_value = preds[i]
        if min_value is None or min_value > targets[i]:
            min_value = targets[i]
        if max_value is None or max_value < preds[i]:
            max_value = preds[i]
        if max_value is None or max_value < targets[i]:
            max_value = targets[i]

    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([min_value, max_value], [min_value, max_value], c='b')
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.grid(False)
    if save_dir is not None:
        plt.savefig(save_dir + "valid_pred.svg")
    plt.show()
