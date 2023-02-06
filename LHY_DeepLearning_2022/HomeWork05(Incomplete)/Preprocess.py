# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/12/17 21:57
import os
import random
import re
from pathlib import Path
import sentencepiece as spm


def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace('-', '')  # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s)  # Q2B
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)  # keep punctuation
    s = ' '.join(s.strip().split())
    return s


def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0:  # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # remove by ratio of length
                            if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


def split_data(prefix, src_lang, tgt_lang, data_prefix, train_ratio):
    if (prefix / f'train.clean.{src_lang}').exists() \
            and (prefix / f'train.clean.{tgt_lang}').exists() \
            and (prefix / f'valid.clean.{src_lang}').exists() \
            and (prefix / f'valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
    else:
        line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
        labels = list(range(line_num))
        random.shuffle(labels)
        for lang in [src_lang, tgt_lang]:
            train_f = open(os.path.join(prefix, f'train.clean.{lang}'), 'w')
            valid_f = open(os.path.join(prefix, f'valid.clean.{lang}'), 'w')
            count = 0
            for line in open(f'{data_prefix}.clean.{lang}', 'r'):
                if labels[count] / line_num < train_ratio:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                count += 1
            train_f.close()
            valid_f.close()

def Subword_Units(prefix):
    src_lang = 'en'
    tgt_lang = 'zh'
    prefix = Path('./data')
    vocab_size = 8000
    if (prefix / f'spm{vocab_size}.model').exists():
        print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
        return
    else:
        spm.SentencePieceTrainer.train(
            input=','.join([f'{prefix}/train.clean.{src_lang}',
                            f'{prefix}/valid.clean.{src_lang}',
                            f'{prefix}/train.clean.{tgt_lang}',
                            f'{prefix}/valid.clean.{tgt_lang}']),
            model_prefix=prefix / f'spm{vocab_size}',
            vocab_size=vocab_size,
            character_coverage=1,
            model_type='unigram',  # 'bpe' works as well
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )
        spm_model = spm.SentencePieceProcessor(model_file=str(prefix / f'spm{vocab_size}.model'))
        in_tag = {
            'train': 'train.clean',
            'valid': 'valid.clean',
            'test': 'test.raw.clean',
        }
        for split in ['train', 'valid', 'test']:
            for lang in [src_lang, tgt_lang]:
                out_path = prefix / f'{split}.{lang}'
                if out_path.exists():
                    print(f"{out_path} exists. skipping spm_encode.")
                else:
                    with open(prefix / f'{split}.{lang}', 'w', encoding="UTF-8") as out_f:
                        with open(prefix / f'{in_tag[split]}.{lang}', 'r', encoding="UTF-8") as in_f:
                            for line in in_f:
                                line = line.strip()
                                tok = spm_model.encode(line, out_type=str)
                                print(' '.join(tok), file=out_f)