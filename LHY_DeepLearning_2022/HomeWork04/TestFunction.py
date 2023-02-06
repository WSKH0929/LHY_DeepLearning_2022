# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 13:27
import numpy as np
import torch


def test(test_loader,mapping, model, device):
    model.eval()
    # 预测并记录结果
    results = [["Id", "Category"]]
    for feat_paths, mels in test_loader:
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])
    return results