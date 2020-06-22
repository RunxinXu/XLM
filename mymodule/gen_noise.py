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
    count = len(clean_data)
    print(count)
    for line_s, line_t in tqdm(zip(shuffle(clean_data["src"]),
                                   shuffle(clean_data["tgt"]))):
        misaligned.append({
            "src": line_s,
            "tgt": line_t,
            "label": "0"
        })
        count -= 1
        if count == 0:
            'break'
            break
    misaligned = shuffle(misaligned)
    return pd.DataFrame(misaligned)


def _gen_short_segments(clean_data, n_tokens=[2,3]):
    short_segments = []
    count = len(clean_data) // 3

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
    count = len(clean_data)

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
    count = len(clean_data)

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

# 需要先分词好！！！！

base_path = '/mnt/cephfs_new_wj/bytetrans/runxindidi/wmt_filter/km/parallel'
clean_data_src = os.path.join(base_path, 'filter.en-km.en.tok')
clean_data_trg = os.path.join(base_path, 'filter.en-km.km.tok')
path = '/mnt/cephfs_new_wj/bytetrans/runxindidi/wmt_filter/XLM/mymodule/en-km/second/data'
output_src = os.path.join(path, 'all.en.tok')
output_trg = os.path.join(path, 'all.km.tok')
output_label = os.path.join(path, 'all.label')

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
