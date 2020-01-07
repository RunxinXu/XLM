# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import random
import os


def load_clean_data(file_src, file_tgt):
    data = []
    with open(file_src) as fs, open(file_tgt) as ft:
        for line_s, line_t in tqdm(zip(fs, ft)):
            line_s = line_s.split()
            line_t = line_t.split()
            data.append({
                "src": line_s,
                "tgt": line_t,
                "label": "1"
            })
    return pd.DataFrame(data)

def gen_misaligned(clean_data):
    misaligned = []
    count = len(clean_data) / 2

    for line_s, line_t in tqdm(zip(shuffle(clean_data["src"]),
                                   shuffle(clean_data["tgt"]))):
        misaligned.append({
            "src": line_s,
            "tgt": line_t,
            "label": "0"
        })
        count -= 1
        if count == 0:
            break
    misaligned = shuffle(misaligned)
    return pd.DataFrame(misaligned)


def _gen_short_segments(clean_data, n_tokens=[2,3]):
    short_segments = []
    count = len(clean_data) / 5

    for line_s, line_t in tqdm(zip(shuffle(clean_data["src"]),
                                    shuffle(clean_data["tgt"]))):
        len_s = len(line_s)
        len_t = len(line_t)
        n = n_tokens[random.randint(0,1)]
        sizes = [(n, n), (n, len_t), (len_s, n)]
        s = sizes[random.randint(0,2)]

        short_segments.append({
            "src": line_s[:s[0]],
            "tgt": line_t[:s[1]],
            "label": "0"
        })

        count -= 1
        if count == 0:
            break
    return pd.DataFrame(short_segments)


def gen_misordered(clean_data):
    misordered = []
    count = len(clean_data) / 2

    for line_s, line_t in tqdm(zip(shuffle(clean_data["src"]),
                                   shuffle(clean_data["tgt"]))):
        tmp_s_s = shuffle(line_s)
        tmp_t_s = shuffle(line_t)
        data = [(tmp_s_s, tmp_t_s), (tmp_s_s, line_t), (line_s, tmp_t_s)]
        index = random.randint(0,2)
        misordered.append({
            "src": data[index][0],
            "tgt": data[index][1],
            "label": "0"
        })
        
        count -= 1
        if count == 0:
            break
    misordered = shuffle(misordered)
    return pd.DataFrame(misordered)


def gen_wrong_language(clean_data):
    wrong_language = []
    count = len(clean_data) / 2

    for line_s, line_t \
            in tqdm(zip(
        shuffle(clean_data["src"]),
        shuffle(clean_data["tgt"]))):
        data = [
            (line_s, line_s),
            (line_t, line_t),
            (line_t, line_s),
        ]

        index = random.randint(0, 2)
        wrong_language.append({
            "src": data[index][0],
            "tgt": data[index][1],
            "label": "0"
        })
        
        count -= 1
        if count == 0:
            break

    wrong_language = shuffle(wrong_language)
    return pd.DataFrame(wrong_language)

clean_data_src = 'en2zh/en.pos'
clean_data_trg = 'en2zh/zh.pos'
output_src = 'en2zh/en.train'
output_trg = 'en2zh/zh.train'
output_label = 'en2zh/en2zh_label.train'

clean_data = load_clean_data(clean_data_src, clean_data_trg)
misaligned = gen_misaligned(clean_data)
misordered = gen_misordered(clean_data)
short = _gen_short_segments(clean_data)
wrong = gen_wrong_language(clean_data)
data = pd.concat([clean_data, misaligned, misordered, short, wrong])
data = shuffle(data).reset_index(drop=True)

with open(output_src, 'w') as f:
    for text in data['src']:
        f.write(" ".join(text) + os.linesep)
with open(output_trg, 'w') as f:
    for text in data['tgt']:
        f.write(" ".join(text) + os.linesep)
with open(output_label, 'w') as f:
    for label in data['label']:
        f.write(label+os.linesep)

# from tokenizer import ChineseTokenizer, EnglishTokenizer
# def gen_noise_for_commoncrawl(src, trg, result_src, result_trg, tokenize=False, include_clean=True):
#     if tokenize == True:
#         zh_tokenizer = ChineseTokenizer()
#         en_tokenizer = EnglishTokenizer()
#     file_src = src
#     file_tgt = trg
#     data = []
#     count = 0
#     with open(file_src) as fs, open(file_tgt) as ft:
#         for line_s, line_t in tqdm(zip(fs, ft)):
#             if tokenize == True:
#                 line_s = zh_tokenizer.tokenize(line_s)
#                 line_t = en_tokenizer.tokenize(line_t)
#             line_s = line_s.split()
#             line_t = line_t.split()
#             data.append({
#                 "src": line_s,
#                 "tgt": line_t,
#             })
#             count += 1
#             if count >= 5000000:
#                 break
#     clean_data = pd.DataFrame(data)

#     # 200W 
#     count = 0
#     misaligned = []
#     for line_s, line_t in tqdm(zip(shuffle(clean_data["src"]),
#                                    shuffle(clean_data["tgt"]))):
#         misaligned.append({
#             "src": line_s,
#             "tgt": line_t,
#         })
#         count += 1
#         if count >= 2000000:
#             break
#     misaligned = pd.DataFrame(misaligned)

#     # 200W
#     count = 0
#     misordered = []
#     for line_s, line_t in tqdm(zip(shuffle(clean_data["src"]),
#                                    shuffle(clean_data["tgt"]))):
#         tmp_s_s = shuffle(line_s)
#         tmp_t_s = shuffle(line_t)
#         data = [(tmp_s_s, tmp_t_s), (tmp_s_s, line_t), (line_s, tmp_t_s)]
#         index = random.randint(0,2)
#         misordered.append({
#             "src": data[index][0],
#             "tgt": data[index][1],
#         })
#         count += 1
#         if count >= 2000000:
#             break
#     misordered = pd.DataFrame(misordered)
#     if include_clean == True:
#         data = pd.concat([clean_data, misaligned, misordered])
#     else:
#         data = pd.concat([misaligned, misordered])
#     data = shuffle(data).reset_index(drop=True)
#     with open(result_src, 'w') as f:
#         for text in data['src']:
#             f.write(" ".join(text) + os.linesep)
#     with open(result_trg, 'w') as f:
#         for text in data['tgt']:
#             f.write(" ".join(text) + os.linesep)

# gen_noise_for_commoncrawl('/mnt/cephfs_new_wj/bytetrans/runxindidi/commoncrawl/origin/train_zh', '/mnt/cephfs_new_wj/bytetrans/runxindidi/commoncrawl/origin/train_en',
#     '/mnt/cephfs_new_wj/bytetrans/runxindidi/commoncrawl/origin_add_noise/train_zh', '/mnt/cephfs_new_wj/bytetrans/runxindidi/commoncrawl/origin_add_noise/train_en')
# gen_noise_for_commoncrawl('/mnt/cephfs_new_wj/bytetrans/runxindidi/train_lm/youdao_zh', '/mnt/cephfs_new_wj/bytetrans/runxindidi/train_lm/youdao_en',
#     '/mnt/cephfs_new_wj/bytetrans/runxindidi/commoncrawl/origin_add_youdao_noise/train_zh', '/mnt/cephfs_new_wj/bytetrans/runxindidi/commoncrawl/origin_add_youdao_noise/train_en', include_clean=False)
