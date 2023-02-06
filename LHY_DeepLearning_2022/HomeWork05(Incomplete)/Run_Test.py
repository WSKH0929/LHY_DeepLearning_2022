# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:34
import csv
import json
import pickle
from torch.utils.data import DataLoader
from HomeWork04.DataSet import *
from HomeWork04.TestFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)
    return feat_paths, torch.stack(mels)


def save_pred(preds, path):
    """ Save predictions to specified file """
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(preds)
        print("Saved successfully: " + path)


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_dir = r"D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch04\Dataset\Dataset"

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
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    test_dataset = InferenceDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=inference_collate_batch)

    # init Model and load saved model's parameter
    speaker_num = len(mapping["id2speaker"])
    model = DNN(n_speakers=speaker_num).to(config['device'])

    ckpt = torch.load(config['model_save_dir'] + "model.pth", map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)

    # test process
    preds = test(test_loader, mapping, model, config['device'])

    # save prediction file to test_pred.csv
    save_pred(preds, config['test_save_dir'] + 'test_pred.csv')
