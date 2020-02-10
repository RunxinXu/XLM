# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, params, dico, mode): 
        self.params = params
        self.dico = dico

        data_path = params.data_path
        src_file = os.path.join(data_path, 'multi.bpe.{}'.format(mode))
        trg_file = os.path.join(data_path, 'multi.bpe.{}'.format(mode))
        lang_file = os.path.join(data_path, 'multi.lang.{}'.format(mode))

        # optional
        label_file = None
        src_raw_text_file = None
        trg_raw_text_file = None

        if mode in ['train', 'valid']:
            label_file = os.path.join(data_path, 'multi.label.{}'.format(mode))
        else:
            src_raw_text_file = os.path.join(data_path, 'multi.{}'.format(mode))
            trg_raw_text_file = os.path.join(data_path, 'multi.{}'.format(mode))

        src = open(src_file, 'r').read().splitlines()
        trg = open(trg_file, 'r').read().splitlines()
        langs = open(lang_file, 'r').read().splitlines()
        
        label = None
        src_raw_text = None
        trg_raw_text = None

        if mode in ['train', 'valid']:
            label = open(label_file, 'r').read().splitlines()
        else:
            src_raw_text = open(src_raw_text_file, 'r').read().splitlines()
            trg_raw_text = open(trg_raw_text_file, 'r').read().splitlines()

        self.data = []
            
        for i in tqdm(range(len(src))):
            src_words = [params.eos_index] + [dico.index(s) for s in src[i].strip().split()] + [params.eos_index]
            trg_words = [params.eos_index] + [dico.index(s) for s in trg[i].strip().split()] + [params.eos_index]

            src_words = torch.tensor(src_words)
            trg_words = torch.tensor(trg_words)
            
            lang1, lang2 = langs[i].split()

            if mode in ['train', 'valid']:
                lab = [int(label[i].strip())]
                lab = torch.tensor(lab)
                src_text = None
                trg_text = None
            else:
                lab = None
                src_text = src_raw_text[i]
                trg_text = trg_raw_text[i]

            self.data.append({
                'src_words': src_words,
                'src_len': len(src_words),
                'trg_words': trg_words,
                'trg_len': len(trg_words),
                'label': lab,
                'src_text': src_text,
                'trg_text': trg_text,
                'lang1': self.params.lang2id[lang1],
                'lang2': self.params.lang2id[lang2],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]

def collate_fn(batch):

    pad_index = 2

    src_words = [sample['src_words'] for sample in batch]
    src_len = [sample['src_len'] for sample in batch]
    trg_words = [sample['trg_words']for sample in batch]
    trg_len = [sample['trg_len'] for sample in batch]

    src_words = nn.utils.rnn.pad_sequence(src_words, padding_value = pad_index).long()
    src_len = torch.tensor(src_len).long()
    trg_words = nn.utils.rnn.pad_sequence(trg_words, padding_value = pad_index).long()
    trg_len = torch.tensor(trg_len).long()

    labels = [sample['label'] for sample in batch]
    if labels[0] is not None:
        labels = torch.tensor(labels).long()
    src_texts = [sample['src_text'] for sample in batch]
    trg_texts = [sample['trg_text'] for sample in batch]

    lang1 = torch.tensor([sample['lang1'] for sample in batch]).long()
    lang2 = torch.tensor([sample['lang2'] for sample in batch]).long()

    return src_words, src_len, trg_words, trg_len, labels, src_texts, trg_texts, lang1, lang2


if __name__ == '__main__':
    dataset = MyDataset('../processed_data/test', opt, mode='test')
    data_loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)
        input()
    