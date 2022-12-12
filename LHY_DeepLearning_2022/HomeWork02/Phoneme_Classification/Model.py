# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:19
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=1):
        super(DNN, self).__init__()

        # Improve: use the nice network structure
        self.layers = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

        # CrossEntropyLoss
        # 注意：CrossEntropyLoss 内部结合了One-Hot编码和SoftMax归一化，所以不需要我们在输出时手动归一化，也不需要传入独热编码给CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def cal_loss(self, pred, target):
        """ Calculate loss """
        # TODO: you may implement L1/L2 regularization here

        # Improve: L1 regularization
        # l1_lambda = 0.001
        # l1_loss = 0
        # for name, w in self.layers.named_parameters():
        #     if 'weight' in name:
        #         l1_loss += l1_lambda * torch.norm(w, p=1)
        #
        # return self.criterion(pred, target) + l1_loss

        # Improve: L2 regularization
        # l2_lambda = 0.001
        # l2_loss = 0
        # for name, w in self.layers.named_parameters():
        #     if 'weight' in name:
        #         l2_loss += l2_lambda * torch.norm(w, p=2)

        return self.criterion(pred, target)
