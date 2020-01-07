# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class MyDataset(Dataset):
    def __init__(self, params, dico, mode): 
        self.params = params
        self.dico = dico

        data_path = params.data_path
        src_file = os.path.join(data_path, '{}.bpe.{}'.format(params.src_lang, mode))
        trg_file = os.path.join(data_path, '{}.bpe.{}'.format(params.trg_lang, mode))
        label_file = os.path.join(data_path, '{}2{}_label.{}'.format(params.src_lang, params.trg_lang, mode))

        src = open(src_file, 'r').read().splitlines()
        trg = open(trg_file, 'r').read().splitlines()
        label = open(label_file, 'r').read().splitlines()
        
        assert len(src) == len(trg)
        assert len(trg) == len(label)
        self.data = []
            
        for i in range(len(src)):
            src_words = [params.eos_index] + [dico.index(s) for s in src[i].strip().split()] + [params.eos_index]
            trg_words = [params.eos_index] + [dico.index(s) for s in trg[i].strip().split()] + [params.eos_index]
            lab = [int(label[i].strip())]

            src_words = torch.tensor(src_words)
            trg_words = torch.tensor(trg_words)
            lab = torch.tensor(lab)
            self.data.append({
                'src_words': src_words,
                'src_len': len(src_words),
                'trg_words': trg_words,
                'trg_len': len(trg_words),
                'label': lab
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
    labels = [sample['label'] for sample in batch]

    src_words = nn.utils.rnn.pad_sequence(src_words, padding_value = pad_index).long()
    src_len = torch.tensor(src_len).long()
    trg_words = nn.utils.rnn.pad_sequence(trg_words, padding_value = pad_index).long()
    trg_len = torch.tensor(trg_len).long()
    labels = torch.tensor(labels).long()

    return src_words, src_len, trg_words, trg_len, labels


if __name__ == '__main__':
    dataset = MyDataset('../processed_data/test', opt, mode='test')
    data_loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    for batch in data_loader:
        print(batch)
        input()
    