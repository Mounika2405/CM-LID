import ast
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader

class CMDataset():
    
    def __init__(self, data_path):
        self.data_path = data_path
        with open(self.data_path, 'r') as f:
            d = f.read()
        self.data_dict = ast.literal_eval(d) #To consider d as type dictionary 
    
    def __len__(self):
        return len(self.data_dict['text'])

    def __getitem__(self, index):

        x = torch.tensor(self.data_dict['text'][index], dtype=torch.float)
        y = torch.tensor(self.data_dict['labels'][index])
        
        return x, y

def collate_pad(batch):
    
    text = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # print('text[0]',text[0].shape)
    # sizes = []
    # for t in text:
    #     sizes.append(t.shape[0])
    # lengths = [len(text)] * max(sizes)

    # text = pad_sequence(text, batch_first=True, padding_value=20) #check #padval
    # labels = pad_sequence(labels, batch_first=True, padding_value=0.5) #check #padval

    return text, labels

def generate_batches(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader







