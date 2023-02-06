# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2023/1/7 21:06

import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms


class HumanDataset(TensorDataset):
    # TensorDataset 支持transform转换
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            # permute 维度交换 交换前[100000, 64, 64, 3], 交换后[100000, 3, 64, 64]
            self.tensors = tensors.permute(0, 3, 1, 2)

        # 加入transform的意义是将pixel从[0, 255]转化为[-1, 1]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x:x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x/255. - 1.)
        ])

    def __getitem__(self, item):
        img = self.tensors[item]
        if self.transform:
            # 将pixel从[0, 255]转化为[-1, 1]
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.tensors)
