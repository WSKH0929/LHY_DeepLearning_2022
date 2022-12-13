# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 13:27
import numpy as np
import torch


def test(test_loader, model, device):
    model.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in test_loader:
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()
    return prediction
