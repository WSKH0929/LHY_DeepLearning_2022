# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
import torch
from torch.utils.data import Dataset, DataLoader


# 定义dataset类
class COVID19Dataset(Dataset):
    """ Dataset for loading and preprocessing the COVID19 dataset """

    def __init__(self,
                 data,
                 feats,
                 mode='train'):
        self.mode = mode

        if mode == 'test':
            # Testing data
            # data: 893 x feature_dim
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/val sets)
            # data: 2700 x (feature_dim + target_dim)
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & val sets
            indices = []
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'val':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'val']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


def prep_dataloader(data, mode, batch_size, feats, n_jobs=0):
    """ Generates a dataset, then is put into a dataloader. """
    dataset = COVID19Dataset(data, feats, mode=mode)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)  # Construct dataloader
    return dataloader
