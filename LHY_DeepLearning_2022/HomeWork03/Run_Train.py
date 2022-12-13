# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15
import pickle

from torch.utils.data import DataLoader
from HomeWork03.TrainFunction import *
from HomeWork03.DataSet import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path
import torchvision.transforms as transforms
from functools import partial

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_path = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch03\food11\food11'

    # transforms 是常用的图像预处理函数 可以进行一些诸如裁剪、缩放等操作提高泛化能力 防止过拟合  即Data Augmentation过程
    transforms_list = [
        partial(transforms.RandomVerticalFlip, p=1),  # 随机水平旋转
        partial(transforms.RandomHorizontalFlip, p=1),  # 随机水平旋转
        partial(transforms.ColorJitter, brightness=0.5),  # 调整亮度
        partial(transforms.CenterCrop, 128)  # 不做任何处理
    ]
    valid_transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        # 对图片随机产生一种影响
        # random.choice(transforms_list)(),
        transforms.ToTensor(),
    ])

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 20,  # Number of epochs.
        'batch_size': 64,
        'optimizer': 'Adam',
        'optim_hyper_paras': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.0003,  # learning rate of optimizer
            'weight_decay': 1e-5,
        },
        'early_stop': 200,  # If model has not improved for this many consecutive epochs, stop training.

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

    # 由于数据集已经划分好了，直接读取数据集
    train_set = FoodDataset(os.path.join(data_path, 'training'), train_transforms, max_size=None)
    valid_set = FoodDataset(os.path.join(data_path, 'validation'), valid_transforms, max_size=None)
    print(f'train_set.size: {len(train_set)} , valid_set.size: {len(valid_set)}')

    # num_worker 是主动将batch加载进内存的workers数,一般设置与CPU核心数一致
    # pin_memory 锁页内存 将Tensor从内存移动到GPU的速度会变快，高端设备才行
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    # init Model (Construct model and move to device)
    model = CNN().to(config['device'])

    # train process
    model_loss_record = train(train_loader, valid_loader, model, config, config['device'])

    # plot and save learning curve and valid_pred image
    plot_learning_curve(model_loss_record, title='CNN', save_dir=config['train_save_dir'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
