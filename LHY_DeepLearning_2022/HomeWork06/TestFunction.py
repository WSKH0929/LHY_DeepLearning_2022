# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 13:27
import logging
import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable


def test(G, config):
    G.eval()
    z_samples = Variable(torch.randn(100, config["z_dim"])).cuda()
    FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=FORMAT,
                        datefmt='%Y-%m-%d %H:%M')

    f_imgs_sample = (G(z_samples).data + 1) / 2.0
    filename = os.path.join(config['test_save_dir'], f'test.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    logging.info(f'Save some samples to {filename}.')
    # Show some images during training.
    grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
