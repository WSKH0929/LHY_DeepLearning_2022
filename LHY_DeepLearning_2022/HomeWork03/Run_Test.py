# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:34
import json
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from HomeWork03.DataSet import *
from HomeWork03.Model import CNN
from HomeWork03.TestFunction import *
from Utils.MyDLUtil import *
from pathlib import Path


def save_pred(pred, path):
    """ Save predictions to specified file """
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1, len(test_set) + 1)]
    df["Category"] = pred
    df.to_csv(path, index=False)
    print("Saved successfully: " + path)


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_path = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch03\food11\food11'

    test_transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), ])

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
    test_set = FoodDataset(os.path.join(data_path, 'validation'), test_transforms, max_size=None)
    print(f'test_set.size: {len(test_set)}')
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # init Model and load saved model's parameter
    model = CNN().to(config['device'])

    ckpt = torch.load(config['model_save_dir'] + "model.pth", map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)

    # test process
    preds = test(test_loader, model, config['device'])

    # save prediction file to test_pred.csv
    save_pred(preds, config['test_save_dir'] + 'test_pred.csv')
