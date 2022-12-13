# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


# 定义Dataset
class FoodDataset(Dataset):

    def __init__(self, path, transform, max_size=None, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')])  # 读取所有文件的名字
        if max_size is not None and max_size < len(self.files):
            self.files = [self.files[i] for i in range(max_size)]
        if files:
            self.files = files
        self.transform = transform

    def __getitem__(self, item):
        fname = self.files[item]
        img = Image.open(fname)
        img = self.transform(img)
        try:
            label = int(fname.split("\\")[-1].split('_')[0])
        except:
            label = -1
        return img, label

    def __len__(self):
        return len(self.files)
