
from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F

logger = getLogger()

class MyModel(object):
    def __init__(self, embedder, params):
        
        self.embedder = embedder
        self.proj = nn.Sequential(*[
                    nn.Dropout(params.dropout),
                    nn.Linear(self.embedder.out_dim, 2)
                ])
        

