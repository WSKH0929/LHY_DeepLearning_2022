# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 13:27
import numpy as np
import torch


def test(test_loader, model, device):
    pred = np.array([], dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate((test_loader)):
            features = batch
            features = features.to(device)
            outputs = model(features)
            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
    return pred
