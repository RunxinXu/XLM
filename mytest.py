import os
import argparse

from mymodule.mymodel import MyModel
from mymodule.mytask import MyTask

from src.utils import bool_flag, initialize_exp
from src.model.embedder import SentenceEmbedder
import torch

# parse parameters
parser = argparse.ArgumentParser(description='Train on GLUE or XNLI')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")

parser.add_argument("--mode", type=str, default="train",
                    help="train or test")
parser.add_argument("--save_model", type=str, default="",
                    help="save model path")
parser.add_argument("--model_path", type=str, default="",
                    help="Model location")
parser.add_argument("--test_result_path", type=str, default="",
                    help="test_result_path")
       
parser.add_argument("--src_lang", type=str, default="",
                    help="src lang")
parser.add_argument("--trg_lang", type=str, default="",
                    help="trg lang")
            
# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--min_count", type=int, default=0,
                    help="Minimum vocabulary count")

# batch parameters
parser.add_argument("--max_len", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                    help="Embedder (pretrained model) optimizer")
parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                    help="Projection (classifier) optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")

# evaluate  每过eval_interval * batch_size 个数据 
parser.add_argument("--eval_interval", type=int, default=10000,
                    help="eval interval")
# maintain超过多少score就写出
parser.add_argument("--flush_frequency", type=int, default=100000,
                    help="eval interval")

# parse parameters
params = parser.parse_args()

# check parameters
assert os.path.isdir(params.data_path)
assert os.path.isdir(params.dump_path)
assert os.path.isfile(params.model_path)
assert params.mode in ['train', 'test']
if params.mode == 'train':
    assert params.save_model != ""
    if not os.path.exists(params.save_model):
        os.makedirs(params.save_model)
else:
    assert params.test_result_path != ""

# train(fine-tune)的话，model_path指向pre-train model
# 之后构造一个MyModel
# test的话，model_path指向一个MyModel对象 直接load
if params.mode == 'train':
    # reload pretrained model
    embedder = SentenceEmbedder.reload(params.model_path, params)
    my_model = MyModel(embedder, params)
else:
    my_model = torch.load(params.model_path)

# reload langs from pretrained model
params.n_langs = my_model.embedder.pretrain_params['n_langs']
params.id2lang = my_model.embedder.pretrain_params['id2lang']
params.lang2id = my_model.embedder.pretrain_params['lang2id']

if params.max_vocab > 1:
    my_model.embedder.dico.max_vocab(params.max_vocab)
if params.min_count > 0:
    my_model.embedder.dico.min_count(params.min_count)

params.bos_index = my_model.embedder.dico.bos_index # 0
params.eos_index = my_model.embedder.dico.eos_index # 1
params.pad_index = my_model.embedder.dico.pad_index # 2
params.unk_index = my_model.embedder.dico.unk_index # 3

# initialize the experiment
logger = initialize_exp(params)

task = MyTask(my_model, params)
task.run()


