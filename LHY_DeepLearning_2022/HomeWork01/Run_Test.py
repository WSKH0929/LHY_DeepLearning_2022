# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:34
import csv
import json
import os
import pickle

from HomeWork01.DataSet import *
from HomeWork01.TestFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


def save_pred(preds, path):
    """ Save predictions to specified file """
    with open(path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
        print("Saved successfully: " + path)


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

    # test data path
    test_data_path = './data/covid.test.csv'

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
    test_data = read_data(test_data_path)

    # Improve: z-score standard scaler for features
    test_data = z_score_standard_scaler(test_data)

    # feature selection
    feats = config['feats']

    # convert data to DataLoader
    test_loader = prep_dataloader(test_data.copy(), 'test', config['batch_size'], feats)

    # init Model and load saved model's parameter
    model = DNN(test_loader.dataset.dim).to(config['device'])
    ckpt = torch.load(config['model_save_dir'] + "model.pth", map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)

    # test process
    preds = test(test_loader, model, config['device'])

    # save prediction file to test_pred.csv
    save_pred(preds, config['test_save_dir'] + 'test_pred.csv')
