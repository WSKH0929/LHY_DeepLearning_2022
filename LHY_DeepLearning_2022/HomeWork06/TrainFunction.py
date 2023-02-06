# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.optim import Optimizer
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from HomeWork06.DataSet import *
from HomeWork06.Model import *


def train(train_loader, config, device, G, D):
    # Setup optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(0.5, 0.999))

    # loss function
    loss = nn.BCELoss()

    FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=FORMAT,
                        datefmt='%Y-%m-%d %H:%M')

    z_samples = Variable(torch.randn(100, config["z_dim"])).cuda()

    loss_record = {"G": [], "D": []}
    steps = 0
    for e, epoch in enumerate(range(config["n_epoch"])):
        progress_bar = tqdm(train_loader)
        progress_bar.set_description(f"Epoch {e + 1}")
        for i, data in enumerate(progress_bar):
            imgs = data.cuda()
            bs = imgs.size(0)

            # *********************
            # *    Train D        *
            # *********************
            z = Variable(torch.randn(bs, config["z_dim"])).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()

            # Discriminator forwarding
            r_logit = D(r_imgs)
            f_logit = D(f_imgs)

            """
            NOTE FOR SETTING DISCRIMINATOR LOSS:

            GAN: 
                loss_D = (r_loss + f_loss)/2
            WGAN: 
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
            WGAN-GP: 
                gradient_penalty = gp(r_imgs, f_imgs)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
            """
            # Loss for discriminator
            r_loss = loss(r_logit, r_label)
            f_loss = loss(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # Discriminator backwarding
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            """
            NOTE FOR SETTING WEIGHT CLIP:

            WGAN: below code
            """
            # for p in D.parameters():
            #     p.data.clamp_(-config["clip_value"], config["clip_value"])

            # *********************
            # *    Train G        *
            # *********************
            if steps % config["n_critic"] == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, config["z_dim"])).cuda()
                f_imgs = G(z)

                # Generator forwarding
                f_logit = D(f_imgs)

                """
                NOTE FOR SETTING LOSS FOR GENERATOR:

                GAN: loss_G = loss(f_logit, r_label)
                WGAN: loss_G = -torch.mean(D(f_imgs))
                WGAN-GP: loss_G = -torch.mean(D(f_imgs))
                """
                # Loss for the generator.
                loss_G = loss(f_logit, r_label)

                # Generator backwarding
                G.zero_grad()
                loss_G.backward()
                opt_G.step()

            loss_record["G"].append(loss_G.item())
            loss_record["D"].append(loss_D.item())

            if steps % 10 == 0:
                progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
            steps += 1

        # 验证生成器效果，输出图片
        G.eval()
        f_imgs_sample = (G(z_samples).data + 1) / 2.0
        filename = os.path.join(config['log_dir'], f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        logging.info(f'Save some samples to {filename}.')

        # Show some images during training.
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

        G.train()

        # Save the checkpoints.
        torch.save(G.state_dict(), os.path.join(config['model_save_dir'], f'G_{e}.pth'))
        torch.save(D.state_dict(), os.path.join(config['model_save_dir'], f'D_{e}.pth'))

    logging.info('Finish training')
    return loss_record
