# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:19
import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()

        # Improve: use the nice network structure
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        """
        python call函数会调用forward 所以model.(data) = model.forward(data)
        :param x:
        :return:
        """
        """ Given input of size (batch_size x input_dim), compute output of the network """
        return self.layers(x).squeeze(1)

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
        l2_lambda = 0.001
        l2_loss = 0
        for name, w in self.layers.named_parameters():
            if 'weight' in name:
                l2_loss += l2_lambda * torch.norm(w, p=2)

        return self.criterion(pred, target) + l2_loss