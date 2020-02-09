import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from collections import deque
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
class CRITRICNet(nn.Module):
    def __init__(self,embed_dim,vocab_size,hidden_dim,embed_matrix):
        super(CRITRICNet,self).__init__()
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.emb=nn.Embedding(vocab_size,embed_dim)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,bidirectional=True,batch_first=True)
        self.dropout=nn.Dropout(0.2)
        self.out=nn.Linear(hidden_dim*2,4)
        self.emb.weight=nn.Parameter(torch.tensor(embed_matrix,dtype=torch.float32))
        self.sig=nn.Sigmoid()
    def forward(self,seq):
        inputs=self.emb(seq)
        out,_=self.lstm(inputs)
        avg_pool=torch.mean(out,1)
        out= self.sig(self.out(self.dropout(avg_pool)))
        return out