import os
import argparse

from mymodule.mymodel import MyModel
from mymodule.mytask import MyTask

from src.utils import bool_flag, initialize_exp
from src.model.embedder import SentenceEmbedder

# parse parameters
parser = argparse.ArgumentParser(description='Train on GLUE or XNLI')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--save_model", type=str, default="",
                    help="save model path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")

parser.add_argument("--model_path", type=str, default="",
                    help="Model location")
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
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--max_batch_size", type=int, default=0,
                    help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                    help="Embedder (pretrained model) optimizer")
parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                    help="Projection (classifier) optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")

# debug
parser.add_argument("--debug_train", type=bool_flag, default=False,
                    help="Use valid sets for train sets (faster loading)")
parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                    help="Debug multi-GPU / multi-node within a SLURM job")

# parse parameters
params = parser.parse_args()
if params.tokens_per_batch > -1:
    params.group_by_size = True

# check parameters
assert os.path.isdir(params.data_path)
assert os.path.isfile(params.model_path)

# reload pretrained model
embedder = SentenceEmbedder.reload(params.model_path, params)

# reload langs from pretrained model
params.n_langs = embedder.pretrain_params['n_langs']
params.id2lang = embedder.pretrain_params['id2lang']
params.lang2id = embedder.pretrain_params['lang2id']

# initialize the experiment / build sentence embedder
logger = initialize_exp(params)
scores = {}

my_model = MyModel(embedder, params)
if params.max_vocab > 1:
    my_model.embedder.dico.max_vocab(params.max_vocab)
if params.min_count > 0:
    my_model.embedder.dico.min_count(params.min_count)

params.bos_index = my_model.embedder.dico.bos_index
params.eos_index = my_model.embedder.dico.eos_index
params.pad_index = my_model.embedder.dico.pad_index
params.unk_index = my_model.embedder.dico.unk_index
# print(my_model.embedder.dico.bos_index) # 0
# print(my_model.embedder.dico.eos_index) # 1
# print(my_model.embedder.dico.pad_index) # 2
# print(my_model.embedder.dico.unk_index) # 3

task = MyTask(my_model, scores, params)
task.run()


# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from mymodule.mydataset import collate_fn, MyDataset
# dataset = MyDataset(params, my_model.embedder.dico)
# data_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
# for batch in data_loader:
#     print(batch)
#     input()


