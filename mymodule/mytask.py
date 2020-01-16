# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import time
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mymodule.mydataset import collate_fn, MyDataset

from src.optim import get_optimizer
from src.utils import concat_batches, truncate, to_cuda
from tqdm import tqdm

logger = getLogger()

class MyTask:

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.mode = params.mode
        self.best_acc = 0

    def run(self):
        params = self.params

        # load data
        self.data = self.load_data()
        self.dataloader = self.make_dataloader(self.data)

        self.model.embedder.cuda()
        self.model.proj.cuda()

        #  只打分 不train/eval
        if self.mode == 'test':
            self.test()
            return

        # optimizers
        self.optimizer_e = get_optimizer(list(self.model.embedder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.model.proj.parameters(), params.optimizer_p)

        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("XLM - Training epoch %i ..." % epoch)
            self.train()

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
        lang_id2 = params.lang2id[params.trg_lang]

        count = 0

        for sent1, len1, sent2, len2, y in self.dataloader['train']:
            count += 1
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
                logger.info("XLM - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []
            
            if count % params.eval_interval == 0:
                # evaluation
                logger.info("XLM - Evaluating ")
                with torch.no_grad():
                    scores = self.eval()
                    if scores['acc'] > self.best_acc:
                        self.best_acc = scores['acc']
                        torch.save(self.model, os.path.join(params.save_model, 'best_acc_model.pkl'))
                        with open(os.path.join(params.save_model, 'best_acc.note'), 'a') as f:
                            f.write(str(self.best_acc)+'\n')
                    with open(os.path.join(params.save_model, 'acc.note'), 'a') as f:
                            f.write(str(scores['acc'])+'\n')
                self.model.embedder.train()
                self.model.proj.train()

    def eval(self):
        params = self.params
        self.model.embedder.eval()
        self.model.proj.eval()

        lang_id1 = params.lang2id[params.src_lang]
        lang_id2 = params.lang2id[params.trg_lang]

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
        scores = {}
        scores['acc'] = acc
        return scores

    def test(self):

        params = self.params
        self.model.embedder.eval()
        self.model.proj.eval()

        lang_id1 = params.lang2id[params.src_lang]
        lang_id2 = params.lang2id[params.trg_lang]

        proba_result = []

        with torch.no_grad():

            for sent1, len1, sent2, len2, _  in tqdm(self.dataloader['test']):
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
                x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)

                # forward
                a = self.model.embedder.get_embeddings(x, lengths, positions, langs)
                output = self.model.proj(a)
                proba = F.softmax(output, 1)[:,1] 

                proba_result.extend(proba.cpu().numpy())

                if len(proba_result) > params.flush_frequency:
                    logger.info("write out score...")
                    with open(params.test_result_path, 'a') as f:
                        for score in proba_result:
                            f.write(str(score)+os.linesep)
                        proba_result = []

            # 最后记得写出来剩下的
            logger.info("write out score...")
            with open(params.test_result_path, 'a') as f:
                for score in proba_result:
                    f.write(str(score)+os.linesep)
                proba_result = []

    def load_data(self):
        dataset = {}
        if self.mode == 'train':
            dataset['train'] = MyDataset(self.params, self.model.embedder.dico, 'train')
            dataset['valid'] = MyDataset(self.params, self.model.embedder.dico, 'valid')
        else:
            dataset['test'] = MyDataset(self.params, self.model.embedder.dico, 'test')
        return dataset
    
    def make_dataloader(self, data):
        data_loader = {}
        if self.mode == 'train':
            data_loader['train'] = DataLoader(data['train'], batch_size=self.params.batch_size, shuffle=True, collate_fn=collate_fn)
            data_loader['valid'] = DataLoader(data['valid'], batch_size=self.params.batch_size, shuffle=False, collate_fn=collate_fn)
        else:
            data_loader['test'] = DataLoader(data['test'], batch_size=self.params.batch_size, shuffle=False, collate_fn=collate_fn)
        return data_loader

