# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/12 11:05
import json
import random
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


class SpeakDataset(Dataset):
    def __init__(self, data_dir, segment_len=10):
        self.data_dir = data_dir
        self.segment_len = segment_len  # 为了提高训练效率，将数据分割

        # 加载mapping数据 里面是speaker2id、id2speaker的信息
        mapping_path = Path(data_dir) / 'mapping.json'
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']

        # 加载元数据
        metadata_path = Path(data_dir) / 'metadata.json'
        meta_data = json.load(metadata_path.open())['speakers']

        # 获取speaker的数量 utterance 不同的录音片段
        self.speaker_num = len(meta_data.keys())
        self.data = []
        for speaker in meta_data.keys():
            for utterances in meta_data[speaker]:
                self.data.append([utterances['feature_path'], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feat_path, speaker = self.data[item]
        # 读取数据
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # 切割数据
        if len(mel) > self.segment_len:
            # 随机选择切割点 取一段长度为segment_len的数据
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        speaker = torch.FloatTensor([speaker]).long()
        speaker = torch.squeeze(speaker, dim=0)
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num
