# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:19
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            # nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=0...)
            # 默认kernel是正方形 所以下面这行代码代表 一共 3*3*3的kernel 有64个 padding为1代表上下左右各填充一行
            # 输入的img 是 128*128*3 ， 经过padding变成 130*130*3 所以经过kernel卷积之后 变成 128*128*64
            # nn.BatchNorm2d(num_features, eps=1e-5, momentum==0.1,affine=True,...) 对数据做归一化处理，使其分布均匀，防止梯度消失
            # nn.MaxPool2d(kernel_size, stride, padding,...)最大池化，减小图像尺寸，提高训练速度,变成64 * 64 * 64
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        # 全连接层 FullConnected Layer
        self.fc = nn.Sequential(
            # 输入是512 * 4 * 4 ,把这个Tensor拉直成一个向量，作为输入
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

        # CrossEntropyLoss
        # 注意：CrossEntropyLoss 内部结合了One-Hot编码和SoftMax归一化，所以不需要我们在输出时手动归一化，也不需要传入独热编码给CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()

    # forward相当于定义训练过程
    def forward(self, x):
        out = self.cnn(x)
        # view相当于reshape 把输出拉成一个vector, -1代表自动计算相应长度保证总体元素个数不变
        # out.size()[0]是batch大小 相当于将batch中的每一个sample的特征拉成一个vector 用作全连接层的输入 也可以用flatten函数
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
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
