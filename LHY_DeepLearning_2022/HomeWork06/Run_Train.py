# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15

import gc
import pickle
from torch.utils.data import DataLoader
from HomeWork06.DataSet import *
from HomeWork06.TrainFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch06\crypko_data'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        "model_type": "GAN",
        "batch_size": 64,
        "lr": 1e-4,
        "n_epoch": 10,
        "n_critic": 1,
        "z_dim": 100,
        "workspace_dir": data_dir,  # define in the environment setting

        'model_save_dir': './outputs/models/',  # Your model will be saved here.
        'train_save_dir': './outputs/train/',  # Your model train pred valid data and learning curve will be saved here.
        'test_save_dir': './outputs/test/',  # Your model pred test data will be saved here.
        'log_dir': './outputs/log/',
    }
    print("device:", config['device'])

    # Set seed for reproducibility
    same_seed(config['seed'])

    # create save dir
    Path(config['model_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['train_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    # Get Data
    dataset = get_dataset(os.path.join(config["workspace_dir"], 'faces'))
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)

    # init Model (Construct model and move to device)
    G = Generator(100).cuda()
    D = Discriminator(3).cuda()

    # train process
    loss_record = train(train_loader, config, config['device'], G, D)

    # plot and save learning curve and valid_pred image
    plot_GAN_learning_curve(loss_record, save_dir=config['train_save_dir'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
