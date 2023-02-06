# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/8 21:15

import gc
import pickle
from torch.utils.data import DataLoader
from HomeWork05.DataSet import *
from HomeWork05.Preprocess import *
from HomeWork05.TrainFunction import *
from Model import *
from Utils.MyDLUtil import *
from pathlib import Path

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # data path
    src_lang = 'en'
    tgt_lang = 'zh'
    prefix = "./data"
    data_prefix = f'{prefix}/train_dev.raw'
    test_prefix = f'{prefix}/test.raw'

    # define config
    config = {
        'seed': 929,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'optimizer': 'AdamW',
        'optim_hyper_paras': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 1e-03,  # learning rate of optimizer
        },

        'train_ratio': 0.99,
        "warmup_steps": 2000,
        'total_steps': 70000,

        # cpu threads when fetching & processing data.
        'num_workers': 2,
        # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
        'max_tokens': 8192,
        'accum_steps': 2,

        # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
        'lr_factor': 2.,
        'lr_warmup': 4000,

        # clipping gradient norm helps alleviate gradient exploding
        'clip_norm': 1.0,

        # maximum epochs for training
        'max_epoch': 15,
        'start_epoch': 1,

        # beam size for beam search
        'beam': 5,
        # generate sequences of maximum length ax + b, where x is the source length
        'max_len_a': 1.2,
        'max_len_b': 10,
        # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
        'post_process': "sentencepiece",

        # checkpoints
        'keep_last_epochs': 5,
        'resume': None,  # if resume from checkpoint name (under config.savedir)

        # logging
        'use_wandb': False,

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

    # Preprocess
    clean_corpus(data_prefix, src_lang, tgt_lang)
    clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)
    split_data(prefix, src_lang, tgt_lang, data_prefix, config['train_ratio'])
    Subword_Units(prefix)

    # init Model (Construct model and move to device)
    model = DNN(n_speakers=speck_num).to(config['device'])

    # train process
    model_loss_record = train(train_loader, valid_loader, model, config, config['device'])

    # plot and save learning curve and valid_pred image
    plot_learning_curve(model_loss_record, title='DNN', save_dir=config['train_save_dir'])

    # save config
    with open(config['model_save_dir'] + "config", "wb") as file:
        pickle.dump(config, file)
