import torch
import torch.nn as nn


class LID(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LID, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.l = nn.Linear(57, embedding_dim, bias=False)
        self.h = nn.Linear(self.embedding_dim, 2)
        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        embedding = self.tanh(self.l(x))
        hidden = self.tanh(self.batchnorm(self.h(embedding)))        
        out = self.softmax(hidden) #outputs probabilities for each language
        return out

        
        
