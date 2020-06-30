import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from os import listdir
import numpy as np
from collections import Counter

class CMDataset():
    
    def __init__(self, ngrams_path, labels, vocab_path):
        self.ngrams_path = ngrams_path
        self.labels = pd.read_pickle(labels).lang

        self.vocab_path = vocab_path
        self.ngrams = [] 
        self.vocab_idx = [] 

        for i, j in zip(sorted(listdir(ngrams_path)), sorted(listdir(vocab_path))):
            self.ngrams.append(pd.read_pickle(ngrams_path + i))
            self.vocab_idx.append(pd.read_pickle(vocab_path + j))
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        dim = [240, 1000, 5000, 5000]
        x = preprocess(self.ngrams, index, self.vocab_idx, dim)
        y = np.where(self.labels.iloc[index]=='hin', 0, 1)
        return x, y


def preprocess(ngrams, index, vocab, N):
    vecs = []
    for ng, v, dim in zip(ngrams, vocab, N):
 
        hashed_vec = hash(ng.iloc[index], v, dim)
        vecs.append(hashed_vec)
    vecs = np.hstack(vecs)
    return vecs
        
def hash(ng, v, dim):
    x = np.zeros(dim)
    count = dict(Counter(ng))

    for f in set(ng):
        h = v[f]
        idx = h % dim
                     
#         if ξ(f) == 1:
#             x[idx] += 1
#         else:
        x[idx] += (count[f]/len(ng)) 
    return x

# def collate_pad(batch):
    
#     text = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#     return text, labels

def generate_batches(dataset, batch_size, sampler):

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader







