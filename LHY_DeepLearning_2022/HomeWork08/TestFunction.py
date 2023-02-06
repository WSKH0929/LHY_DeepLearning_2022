# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 13:27
import os

import numpy as np
import pandas as pd
import torch
from torch import nn


def anomaly_detection(test_loader,model, model_type, device, output_dir):
    model.eval()

    eval_loss = nn.MSELoss(reduction='none')

    # 预测结果
    out_file = os.path.join(output_dir, 'prediction.csv')

    # 开始预测
    anomality = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img = data.float().to(device)
            if model_type in ['fcn']:
                img = img.view(img.shape[0], -1)
            output = model(img)
            if model_type in ['vae']:
                output = output[0]
            if model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])

            anomality.append(loss)

    # 将损失值开平方后的结果作为判断的依据
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(19636, 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['score'])
    df.to_csv(out_file, index_label='ID')
