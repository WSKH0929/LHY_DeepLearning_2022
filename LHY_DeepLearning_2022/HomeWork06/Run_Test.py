# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:34
import csv
import json
import pickle
from torch.utils.data import DataLoader
from HomeWork06.DataSet import *
from HomeWork06.TestFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # load and print config
    with open("./outputs/models/config", "rb") as file:
        config = pickle.load(file)
    print("config:")
    print(json.dumps(config, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':')))

    # Set seed for reproducibility
    # same_seed(config['seed'])

    # create test_save_dir
    Path(config['test_save_dir']).mkdir(parents=True, exist_ok=True)

    # init Model and load saved model's parameter
    G = Generator(100).cuda()

    ckpt = torch.load(config['model_save_dir'] + "G_9.pth")  # Load your best model
    G.load_state_dict(ckpt)

    # test process
    test(G, config)
