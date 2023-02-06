# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:19
import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, d_model=80, n_speakers=600, dropout=0.1):
        super().__init__()
        # 设置前置网络pre_net 将feature变为80
        self.pre_net = nn.Linear(40, d_model)
        # TODO：
        # Change Transformer to Conformer
        #
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=2)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_speakers),
        )

        # CrossEntropyLoss
        # 注意：CrossEntropyLoss 内部结合了One-Hot编码和SoftMax归一化，所以不需要我们在输出时手动归一化，也不需要传入独热编码给CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x : 32 * 10 * 40
        out = self.pre_net(x)  # out:32 * 10 * 80
        out = out.permute(1, 0, 2)  # permute 将tensor维度换位  变为10 * 32 * 80
        out = self.encoder_layer(out)  # 10 * 32 * 80
        out = out.transpose(0, 1)  # 32 * 10 * 80
        stats = out.mean(dim=1)  # stats 32 * 80  对batch的每个数据的长度求平均值

        # 预测
        out = self.pred_layer(stats)  # 32 * 600
        return out

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
