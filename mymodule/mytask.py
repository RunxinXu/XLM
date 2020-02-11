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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from mymodule.mydataset import collate_fn, MyDataset

from src.optim import get_optimizer
from src.utils import concat_batches, concat_batches_test, truncate, to_cuda
from tqdm import tqdm

logger = getLogger()

class MyTask:

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.gpu = params.gpu
        self.mode = params.mode

        self.best_acc = 0
        self.global_step = 0

    def run(self):
        # put model in gpu
        self.model.cuda(self.gpu)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], find_unused_parameters=True) 
        cudnn.benchmark = True

        # load data
        self.data = self.load_data()
        self.sampler = self.make_sampler()
        self.dataloader = self.make_dataloader()
        
        if self.mode == 'test':
            self.run_test()
        else:
            self.run_train()

    def run_train(self):
        params = self.params
        # optimizers
        self.optimizer_e = get_optimizer(list(self.model.module.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.model.module.proj.parameters(), params.optimizer_p)

        # criterion 
        self.criterion = nn.CrossEntropyLoss().cuda(self.gpu)

        # train and evaluate the model
        for epoch in range(params.n_epochs):
            self.sampler['train'].set_epoch(epoch) # 因为train的需要shuffle, valid不用

            # update epoch
            self.epoch = epoch

            # training
            logger.info("GPU %i - XLM - Training epoch %i ..." % (self.gpu, epoch))
            self.train()

    def train(self):
        params = self.params
        self.model.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        for sent1, len1, sent2, len2, y, _, _, lang1, lang2 in self.dataloader['train']:
            self.global_step += 1
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang1,
                sent2, len2, lang2,
                params.pad_index,
                params.eos_index,
                reset_positions=True
            )

            bs = len(len1)

            # cuda
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs, gpu=self.gpu)

            # loss
            output = self.model(x, lengths, positions, langs)
            loss = self.criterion(output, y)

            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()
            losses.append(loss.item())

            # log
            if self.global_step % self.params.report_interval == 0:
                logger.info("GPU %i - Epoch %i - Global_step %i - Loss: %.4f" % (self.gpu, self.epoch, self.global_step, sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []
            
            if self.global_step % params.eval_interval == 0:
                if self.gpu == 0:
                    logger.info("XLM - Evaluating")
                    with torch.no_grad():
                        scores = self.eval()
                        if scores['acc'] > self.best_acc:
                            self.best_acc = scores['acc']
                            torch.save(self.model.module, os.path.join(params.save_model, 'best_acc_model.pkl'))
                            with open(os.path.join(params.save_model, 'best_acc.note'), 'a') as f:
                                f.write(str(self.best_acc)+'\n')
                        with open(os.path.join(params.save_model, 'acc.note'), 'a') as f:
                            f.write(str(scores['acc'])+'\n')
                        logger.info("acc - %i " % scores['acc'])
                    self.model.train()

    def eval(self):
        params = self.params
        self.model.eval()

        lang_id1 = params.lang2id[params.src_lang]
        lang_id2 = params.lang2id[params.trg_lang]

        valid = 0
        total = 0

        for sent1, len1, sent2, len2, y, _, _, lang1, lang2 in tqdm(self.dataloader['valid']):
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang1,
                sent2, len2, lang2,
                params.pad_index,
                params.eos_index,
                reset_positions=True
            )

            # cuda
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs, gpu=self.gpu)

            # forward
            output = self.model(x, lengths, positions, langs)
            predictions = output.data.max(1)[1]

            # update statistics
            valid += predictions.eq(y).sum().item()
            total += len(len1)

        # compute accuracy
        acc = 100.0 * valid / total
        scores = {}
        scores['acc'] = acc
        return scores

    def run_test(self):

        params = self.params
        result_path = params.test_result_path + '_{}'.format(self.gpu)
        self.model.eval()

        lang_id1 = params.lang2id[params.src_lang]
        lang_id2 = params.lang2id[params.trg_lang]

        proba_result = []
        src_text_list = []
        trg_text_list = []

        lang1 = self.params.lang2id[self.params.src_lang]
        lang2 = self.params.lang2id[self.params.trg_lang]

        with torch.no_grad():

            for sent1, len1, sent2, len2, _, src_text, trg_text, _, _ in tqdm(self.dataloader['test']):
                sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
                sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
                x, lengths, positions, langs = concat_batches_test(
                    sent1, len1, lang1,
                    sent2, len2, lang2,
                    params.pad_index,
                    params.eos_index,
                    reset_positions=True
                )

                # cuda
                x, lengths, positions, langs = to_cuda(x, lengths, positions, langs, gpu=self.gpu)

                # forward
                output = self.model(x, lengths, positions, langs)
                proba = F.softmax(output, 1)[:,1] 

                proba_result.extend(proba.cpu().numpy())
                src_text_list.extend(src_text)
                trg_text_list.extend(trg_text)
                assert len(proba_result) == len(src_text_list)
                assert len(proba_result) == len(trg_text_list)

                if len(proba_result) > params.flush_frequency:
                    logger.info(" GPU %i - write out score..." % self.gpu)
                    with open(result_path, 'a') as f:
                        for i in range(len(proba_result)):
                            f.write('{}{}{}{}{}'.format(src_text_list[i], params.delimeter, 
                                trg_text_list[i], params.delimeter, str(proba_result[i]))+os.linesep)
                        proba_result = []
                        src_text_list = []
                        trg_text_list = []

            # write out the remainings
            logger.info(" GPU %i - write out score..." % self.gpu)
            with open(result_path, 'a') as f:
                for i in range(len(proba_result)):
                    f.write('{}{}{}{}{}'.format(src_text_list[i], params.delimeter, 
                        trg_text_list[i], params.delimeter, str(proba_result[i]))+os.linesep)
                proba_result = []
                src_text_list = []
                trg_text_list = []

    def load_data(self):
        dataset = {}
        if self.mode == 'train':
            dataset['train'] = MyDataset(self.params, self.model.module.dico, 'train')
            dataset['valid'] = MyDataset(self.params, self.model.module.dico, 'valid')
        else:
            dataset['test'] = MyDataset(self.params, self.model.module.dico, 'test')
        return dataset
    
    def make_dataloader(self):
        data_loader = {}
        if self.mode == 'train':
            data_loader['train'] = DataLoader(self.data['train'], batch_size=self.params.batch_size,
                                               sampler=self.sampler['train'], collate_fn=collate_fn)
            data_loader['valid'] = DataLoader(self.data['valid'], batch_size=self.params.batch_size,
                                               shuffle=False, collate_fn=collate_fn)
        else:
            data_loader['test'] = DataLoader(self.data['test'], batch_size=self.params.batch_size,
                                               sampler=self.sampler['test'], collate_fn=collate_fn)
        return data_loader

    def make_sampler(self):
        sampler = {}
        if self.mode == 'train':
            sampler['train'] = DistributedSampler(self.data['train'], shuffle=True)
            # valid采用single gpu
            # sampler['valid'] = DistributedSampler(self.data['valid'], shuffle=False) 
        else:
            sampler['test'] = DistributedSampler(self.data['test'], shuffle=False)
        return sampler