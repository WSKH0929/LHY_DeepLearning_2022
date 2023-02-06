# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15

import pickle
from torch.utils.data import DataLoader
from HomeWork07.DataSet import *
from HomeWork07.TrainFunction import *
from Utils.MyDLUtil import *
from pathlib import Path
from transformers import BertForQuestionAnswering, BertTokenizerFast


def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    data_dir = r'D:\4 StudyData\Python教程\Pytorch模型数据\Pytorch07\hw7_data'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,

        'num_epoch':1,
        'lr': 1e-04,
        'logging_step':100,

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

    # init Model (Construct model and move to device)
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(config['device'])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # Get Data
    train_questions, train_paragraphs = read_data(os.path.join(data_dir, "hw7_train.json"))
    val_questions, val_paragraphs = read_data(os.path.join(data_dir, "hw7_dev.json"))
    test_questions, test_paragraphs = read_data(os.path.join(data_dir, "hw7_test.json"))

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                          add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in val_questions],
                                        add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                         add_special_tokens=False)

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(val_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    valid_set = QA_Dataset("valid", val_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

    train_batch_size = config['batch_size']

    # Note: Do NOT change batch size of dev_loader / test_loader !
    # Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # train process
    model_acc_record = train(train_loader, valid_loader,val_questions, model, tokenizer, config, config['device'])

    # plot and save learning curve and valid_pred image
    # plot_acc_curve(model_acc_record, title='DNN', save_dir=config['train_save_dir'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
