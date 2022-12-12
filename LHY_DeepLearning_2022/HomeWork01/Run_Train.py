# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15
import csv
import gc
import os
import pickle

from HomeWork01.DataSet import *
from HomeWork01.TrainFunction import *
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

    # training data path
    train_data_path = './data/covid.train.csv'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 3000,  # Number of epochs.
        # Improve: use the nice batch size
        'batch_size': 64,
        # Improve: Use Adam Optimizer
        'optimizer': 'Adam',
        'optim_hyper_paras': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
            # Improve: use the nice learning rate
            'lr': 0.001,  # learning rate of optimizer
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

    # read train data
    train_data = read_data(train_data_path)

    # Improve: z-score standard scaler for features
    train_data[:, :-1] = z_score_standard_scaler(train_data[:, :-1])

    # Improve: Tree-based feature selection
    feats = tree_based_feature_selection(train_data)
    config['feats'] = feats

    # convert data to DataLoader
    train_loader = prep_dataloader(train_data.copy(), 'train', config['batch_size'], feats)
    valid_loader = prep_dataloader(train_data.copy(), 'val', config['batch_size'], feats)

    # init Model (Construct model and move to device)
    model = DNN(train_loader.dataset.dim).to(config['device'])

    # train process
    model_loss, model_loss_record = train(train_loader, valid_loader, model, config, config['device'])

    # plot and save learning curve and valid_pred image
    plot_learning_curve(model_loss_record, title='DNN', save_dir=config['train_save_dir'])
    plot_valid_pred(valid_loader, model, config['device'],
                    save_dir=config['train_save_dir'])  # Show prediction on the validation set

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
