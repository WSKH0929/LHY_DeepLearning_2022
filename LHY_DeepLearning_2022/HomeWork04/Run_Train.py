# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15

import gc
import pickle
from torch.utils.data import DataLoader
from HomeWork04.DataSet import *
from HomeWork04.TrainFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch04\Dataset\Dataset'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'optimizer': 'AdamW',
        'optim_hyper_paras': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 1e-03,  # learning rate of optimizer
        },

        'train_ratio': 0.9,
        "warmup_steps": 2000,
        'total_steps': 70000,

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
    dataset = SpeakDataset(data_dir)
    speck_num = dataset.get_speaker_number()  # 获得演讲者总数

    # 分割数据为训练集0.9和验证集0.1
    train_len = int(config['train_ratio'] * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    train_set, valid_set = random_split(dataset, lengths)

    # 制作train_loader, valid_loader
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                              drop_last=True)  # drop_last 舍弃数据末尾不足一个batch的数据
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                              drop_last=True)  # collate_fn 相当于对每个batch的处理方式

    # init Model (Construct model and move to device)
    model = DNN(n_speakers=speck_num).to(config['device'])

    # train process
    model_loss_record = train(train_loader, valid_loader, model, config, config['device'])

    # plot and save learning curve and valid_pred image
    plot_learning_curve(model_loss_record, title='DNN', save_dir=config['train_save_dir'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
