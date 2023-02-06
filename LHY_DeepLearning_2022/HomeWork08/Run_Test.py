# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:34
import csv
import json
import os
import pickle
from torch.utils.data import DataLoader, SequentialSampler
from HomeWork08.DataSet import HumanDataset
from HomeWork08.TestFunction import anomaly_detection
from Utils.MyDLUtil import *
from pathlib import Path


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)


def save_pred(preds, path):
    """ Save predictions to specified file """
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(preds)
        print("Saved successfully: " + path)


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch08\ml2022spring-hw8\data'

    # load and print config
    with open("./outputs/models/config", "rb") as file:
        config = pickle.load(file)
    print("config:")
    print(json.dumps(config, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':')))

    # Set seed for reproducibility
    same_seed(config['seed'])

    # create test_save_dir
    Path(config['test_save_dir']).mkdir(parents=True, exist_ok=True)

    # read test data
    test_set = np.load(os.path.join(data_dir, 'testingset.npy'), allow_pickle=True)
    test_tensor = torch.tensor(test_set, dtype=torch.float32)
    test_dataset = HumanDataset(test_tensor)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)

    # 加载模型
    checkpoint_path = os.path.join(config['model_save_dir'], f'best_mode_{config["model_type"]}.pt')
    model = torch.load(checkpoint_path)

    # testing
    # 异常检测
    anomaly_detection(test_loader, model, config['model_type'], config['device'], config['test_save_dir'])
