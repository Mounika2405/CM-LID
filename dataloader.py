import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from os import listdir
import numpy as np
from collections import Counter

class CMDataset():
    
    def __init__(self, data_path, vocab_path, is_train=False):
        self.data = pd.read_pickle(data_path)
        self.ngrams = [self.data['unigram'], self.data['bigram'], self.data['trigram'], self.data['4gram']]
        self.labels = self.data.lang
        self.hin_counter = pd.read_pickle('data/hin_dist.pkl')
        self.eng_counter = pd.read_pickle('data/eng_dist.pkl')
        self.num_labels = len(pd.unique(self.labels))
        self.is_train = is_train
        self.vocab_idx = [] 
        
        for i in sorted(listdir(vocab_path)):
            self.vocab_idx.append(pd.read_pickle(vocab_path + i))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        dim = [43, 854, 5000, 5000]
        x = self.preprocess(index, dim)
        y = np.where(self.labels.iloc[index]=='hin', 0, 1)
        return x, y


    def preprocess(self, index, N):
        vecs = []
        for ng, v, dim in zip(self.ngrams, self.vocab_idx, N):
            hashed_vec = hash(ng.iloc[index], v, dim)
            vecs.append(hashed_vec)
        
        if self.is_train:
            prob = np.random.choice([0, 1], p=[0.5, 0.5])
            if prob:
                lang_dist = compute_lang_dist(self.hin_counter, self.eng_counter, self.data.iloc[index].word, self.num_labels)
            else:
                lang_dist = np.array([0.0, 0.0])
        else: 
            lang_dist = compute_lang_dist(self.hin_counter, self.eng_counter, self.data.iloc[index].word, self.num_labels)
        
        vecs.append(lang_dist)
        vecs = np.hstack(vecs)
        return vecs
 

def hash(ng, v, dim):
    x = np.zeros(dim)
    count = dict(Counter(ng))
    for f in set(ng):
        try:
            h = v[f]
        except KeyError:
            h = 0
        idx = h % dim
        x[idx] += (count[f]/len(ng)) 
    return x

def compute_lang_dist(hin_counter, eng_counter, word, num_labels):
    try:
        hin_dist = hin_counter[word]/num_labels
    except KeyError:
        hin_dist = 0.0
    try:
        eng_dist = eng_counter[word]/num_labels
    except KeyError:
        eng_dist = 0.0
    return np.array([hin_dist, eng_dist])

def generate_batches(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader



