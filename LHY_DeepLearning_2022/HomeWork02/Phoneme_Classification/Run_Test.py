# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:34
import csv
import json
import os
import pickle

from torch.utils.data import DataLoader

from HomeWork02.Phoneme_Classification.DataSet import *
from HomeWork02.Phoneme_Classification.TestFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


def save_pred(pred, path):
    """ Save predictions to specified file """
    with open(path, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))
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

    # data path
    feat_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch02\libriphone\feat'
    phone_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch02\libriphone'

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
    x_test = preprocess_data(split='test', feat_dir=feat_dir, phone_path=phone_dir,
                             concat_n_frames=config['concat_n_frames'])

    input_dim = x_test.shape[1]

    test_set = LibriDataset(x_test, None)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    # init Model and load saved model's parameter
    model = DNN(input_dim=input_dim, output_dim=41,
                hidden_layers=config['hidden_layers'],
                hidden_dim=config['hidden_dim']).to(config['device'])

    ckpt = torch.load(config['model_save_dir'] + "model.pth", map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)

    # test process
    preds = test(test_loader, model, config['device'])

    # save prediction file to test_pred.csv
    save_pred(preds, config['test_save_dir'] + 'test_pred.csv')
