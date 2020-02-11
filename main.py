import os
import argparse

from mymodule.mymodel import MyModel
from mymodule.mytask import MyTask

from src.utils import bool_flag, initialize_exp
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

# parse parameters
parser = argparse.ArgumentParser(description='XLM')

# fixed
parser.add_argument("--category", type=int, default=2,
                    help="category")  
                    
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
parser.add_argument("--max_len", type=int, default=128,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch") # batch_size per GPU

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
# train的时候多少global_step播报一次
parser.add_argument("--report_interval", type=int, default=20,
                    help="report interval")
# maintain超过多少score就写出  for single gpu
parser.add_argument("--flush_frequency", type=int, default=10000,
                    help="eval interval")  
parser.add_argument("--delimeter", type=str, default='\t',
                    help="write out delimeter")  

def main():
    # parse parameters
    params = parser.parse_args()
    
    check_params(params)
    
    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(params,))

def main_worker(gpu, params):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23457', world_size=torch.cuda.device_count(), rank=gpu)
    torch.cuda.set_device(gpu) 
    params.gpu = gpu

    # load model
    if params.mode == 'train':
        # reload pretrained model
        my_model = MyModel.reload(params.model_path, params)
    else:
        my_model = torch.load(params.model_path)

    # reload langs from pretrained model
    params.n_langs = my_model.pretrain_params['n_langs']
    params.id2lang = my_model.pretrain_params['id2lang']
    params.lang2id = my_model.pretrain_params['lang2id']

    if params.max_vocab > 1:
        my_model.dico.max_vocab(params.max_vocab)
    if params.min_count > 0:
        my_model.dico.min_count(params.min_count)

    params.bos_index = my_model.dico.bos_index # 0
    params.eos_index = my_model.dico.eos_index # 1
    params.pad_index = my_model.dico.pad_index # 2
    params.unk_index = my_model.dico.unk_index # 3

    # initialize the experiment
    logger = initialize_exp(params)

    task = MyTask(my_model, params)
    task.run()

def check_params(params):
    # check parameters
    assert os.path.isdir(params.data_path)
    assert os.path.isdir(params.dump_path)
    assert os.path.isfile(params.model_path)
    
    assert params.mode in ['train', 'test']
    if params.mode == 'train':
        assert params.save_model != ""
        assert params.src_lang == ""
        assert params.trg_lang == ""
        if not os.path.exists(params.save_model):
            os.makedirs(params.save_model)
    else:
        assert params.test_result_path != ""
        assert params.src_lang != ""
        assert params.trg_lang != ""
        for i in range(torch.cuda.device_count()):
            assert not os.path.isfile(params.test_result_path + '_{}'.format(i))


if __name__ == '__main__':
    main()