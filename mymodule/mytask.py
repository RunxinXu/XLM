# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mymodule.mydataset import collate_fn, MyDataset

from src.optim import get_optimizer
from src.utils import concat_batches, truncate, to_cuda

logger = getLogger()

class MyTask:

    def __init__(self, model, scores, params):
        self.model = model
        self.params = params
        self.scores = scores
        self.best_acc = 0

    def run(self):
        params = self.params

        # load data
        self.data = self.load_data()
        self.dataloader = self.make_dataloader(self.data)

        self.model.embedder.cuda()
        self.model.proj.cuda()

        # optimizers
        self.optimizer_e = get_optimizer(list(self.model.embedder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.model.proj.parameters(), params.optimizer_p)

        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("XNLI - Training epoch %i ..." % epoch)
            self.train()

            # evaluation
            logger.info("XNLI - Evaluating epoch %i ..." % epoch)
            with torch.no_grad():
                scores = self.eval()
                if scores['acc'] > self.best_acc:
                    self.best_acc = scores['acc']
                    torch.save(self.model, os.path.join(params.save_model, 'best_acc_model.pkl'))
                    with open(os.path.join(params.save_model, 'best_acc.note'), 'a') as f:
                        f.write(str(self.best_acc)+'\n')
                self.scores.update(scores)

    def train(self):
        params = self.params
        self.model.embedder.train()
        self.model.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        lang_id1 = params.lang2id[params.src_lang]
        # lang_id2 = params.lang2id[params.trg_lang]
        lang_id2 = params.lang2id['fr']

        for sent1, len1, sent2, len2, y in self.dataloader['train']:
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id1,
                sent2, len2, lang_id2,
                params.pad_index,
                params.eos_index,
                reset_positions=True
            )

            bs = len(len1)

            # cuda
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

            # loss
            output = self.model.proj(self.model.embedder.get_embeddings(x, lengths, positions, langs))
            loss = F.cross_entropy(output, y)

            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())

            # log
            if ns % (100 * bs) < bs:
                logger.info("XNLI - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []

    def eval(self):

        params = self.params
        self.model.embedder.eval()
        self.model.proj.eval()

        scores = OrderedDict({'epoch': self.epoch})

        lang_id1 = params.lang2id[params.src_lang]
        # lang_id2 = params.lang2id[params.trg_lang]
        lang_id2 = params.lang2id['fr']

        valid = 0
        total = 0

        for sent1, len1, sent2, len2, y in self.dataloader['valid']:
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id1,
                sent2, len2, lang_id2,
                params.pad_index,
                params.eos_index,
                reset_positions=True
            )

            # cuda
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

            # forward
            a = self.model.embedder.get_embeddings(x, lengths, positions, langs)
            output = self.model.proj(a)
            predictions = output.data.max(1)[1]

            # update statistics
            valid += predictions.eq(y).sum().item()
            total += len(len1)

        # compute accuracy
        acc = 100.0 * valid / total
        scores['acc'] = acc
        logger.info("Epoch %i - Acc: %.1f%%" % (self.epoch, acc))

        logger.info("__log__:%s" % json.dumps(scores))
        return scores

    def load_data(self):
        dataset = {}
        dataset['train'] = MyDataset(self.params, self.model.embedder.dico, 'train')
        dataset['valid'] = MyDataset(self.params, self.model.embedder.dico, 'valid')
        return dataset
    
    def make_dataloader(self, data):
        data_loader = {}
        data_loader['train'] = DataLoader(data['train'], batch_size=self.params.batch_size, shuffle=True, collate_fn=collate_fn)
        data_loader['valid'] = DataLoader(data['valid'], batch_size=self.params.batch_size, shuffle=False, collate_fn=collate_fn)
        
        return data_loader

