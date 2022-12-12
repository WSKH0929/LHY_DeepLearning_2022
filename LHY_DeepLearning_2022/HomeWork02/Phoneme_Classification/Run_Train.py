# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15
import csv
import gc
import pickle

from torch.utils.data import DataLoader
from HomeWork02.Phoneme_Classification.DataSet import *
from HomeWork02.Phoneme_Classification.TrainFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


def read_data(path):
    """ Read data into numpy arrays """
    with open(path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])[:, 1:].astype(float)
        return data


def tree_based_feature_selection(train_data):
    """ Tree-Based Feature Selector """
    tree = ExtraTreesRegressor()
    tree.fit(train_data[:, :-1], train_data[:, -1])

    select_model = SelectFromModel(tree, prefit=True)
    mask = list(select_model.get_support())
    feats = []
    for i in range(len(mask)):
        if mask[i] is np.True_:
            feats.append(i)

    print("tree_based_feature_selection:", feats)
    return feats


def z_score_standard_scaler(data):
    """ Z-Score Standard Scaler """
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    return data


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    feat_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch02\libriphone\feat'
    phone_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch02\libriphone'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 30,  # Number of epochs.
        'batch_size': 1024,
        'optimizer': 'AdamW',
        'optim_hyper_paras': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 1e-04,  # learning rate of optimizer
        },
        'concat_n_frames': 1,
        'train_ratio': 0.8,
        'hidden_layers': 1,
        'feature_num': 39,
        'hidden_dim': 1024,
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

    # 划分数据集  每个数据39个特征 对应分类为41种
    x_train, y_train = preprocess_data(split='train', feat_dir=feat_dir, phone_path=phone_dir,
                                       concat_n_frames=config['concat_n_frames'], train_ratio=config['train_ratio'],
                                       train_val_seed=config['seed'])
    x_valid, y_valid = preprocess_data(split='val', feat_dir=feat_dir, phone_path=phone_dir,
                                       concat_n_frames=config['concat_n_frames'], train_ratio=config['train_ratio'],
                                       train_val_seed=config['seed'])

    input_dim = x_train.shape[1]

    train_set, valid_set = LibriDataset(x_train, y_train), LibriDataset(x_valid, y_valid)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True)

    # remove raw feature to save memory
    del x_train, y_train, x_valid, y_valid
    gc.collect()

    # init Model (Construct model and move to device)
    model = DNN(input_dim=input_dim, output_dim=41,
                hidden_layers=config['hidden_layers'],
                hidden_dim=config['hidden_dim']).to(config['device'])

    # train process
    model_loss_record = train(train_loader, valid_loader, model, config, config['device'])

    # plot and save learning curve and valid_pred image
    plot_learning_curve(model_loss_record, title='DNN', save_dir=config['train_save_dir'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
