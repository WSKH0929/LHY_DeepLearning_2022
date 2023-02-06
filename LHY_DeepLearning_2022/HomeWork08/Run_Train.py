# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15
import gc
import os
import pickle

from HomeWork08.DataSet import HumanDataset
from HomeWork08.Model import *
from HomeWork08.TrainFunction import train
from Utils.MyDLUtil import *
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch08\ml2022spring-hw8\data'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        'n_epochs': 50,
        'batch_size': 2000,
        'learning_rate': 1e-3,
        'model_type': 'fcn',  # fcn/conv/vae

        'model_save_dir': './outputs/models/',  # Your model will be saved here.
        'train_save_dir': './outputs/train/',  # Your model train pred valid data and learning curve will be saved here.
        'test_save_dir': './outputs/test/',  # Your model pred test data will be saved here.
    }
    print("device:", config['device'])

    # Set seed for reproducibility
    same_seed(config['seed'])

    # create save dir
    Path(config['model_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['train_save_dir']).mkdir(parents=True, exist_ok=True)

    # Get Data
    train_set = np.load(os.path.join(data_dir, "trainingset.npy"), allow_pickle=True)
    train_tensor = torch.from_numpy(train_set)
    train_dataset = HumanDataset(train_tensor)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['batch_size'])

    # 清除内存
    del train_set
    gc.collect()

    # model
    model_classes = {'fcn': FCN_AutoEncoder(), 'conv': Conv_AutoEncoder(), 'vae': VAE()}
    model = model_classes[config['model_type']].to(config['device'])

    # training
    train(train_loader, model, config, config['device'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
