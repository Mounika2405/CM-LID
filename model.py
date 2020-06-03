import torch
import torch.nn as nn


class LID(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, device):
        super(LID, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = device
        
        self.l = nn.Linear(21, embedding_dim, bias=False)
        #Embedding Layer
        # self.embed = nn.Embedding(vocab_size, embedding_dim)
        
        #Hidden Layer
        # self.h = nn.Linear(embedding_dim, hidden_dim)
        self.h = nn.Linear(self.embedding_dim, 2)
        
        #ReLU Layer
        self.relu = nn.ReLU()
        
        #Softmax Layer
        self.softmax = nn.Softmax()

        
    def forward(self, x):
        #Hidden Layer
        
        # embedding = self.embed(x)
        embedding = self.l(x)
        # print('embedding_shape',embedding.shape)
        hidden = self.h(embedding)
        # print('hidden_shape', hidden.shape)
        hidden = self.relu(hidden)
        # print('hidden after relu', hidden.shape)
        
        out = self.softmax(hidden) #outputs probabilities for each language
        
        return out

        
        