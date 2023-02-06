# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
# 定义训练循环
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from HomeWork08.Model import loss_vae


def train(train_loader, model, config, device):
    # 设置迭代器和损失函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 定义相关参数
    best_loss = np.inf
    n_epochs = config['n_epochs']

    # 定义训练过程
    step = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = []

        for data in tqdm(train_loader):
            img = data.float().to(device)  # img [2000, 3, 64, 64]
            if config['model_type'] in ['fcn']:
                img = img.view(img.shape[0], -1)
            output = model(img)

            if config['model_type'] == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)

            train_loss.append(loss.item())

            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # 展示生成的图像
            if step % 100 == 0:
                show_figure(output[0].reshape((3, 64, 64)).permute(1, 2, 0).detach().cpu().numpy(), step,
                            config['train_save_dir'])

        mean_loss = np.mean(train_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, os.path.join(config['model_save_dir'], f'best_mode_{config["model_type"]}.pt'))

        print(f'epoch: {epoch + 1:.0f}/{n_epochs:.0f} , loss: {mean_loss:.4f}')


def show_figure(np_img, step, save_dir):
    plt.figure()
    plt.imshow(np_img)
    plt.axis('on')
    plt.title(f'step {step}')
    plt.savefig(os.path.join(save_dir, f"step_{step}.svg"))
    plt.show()
