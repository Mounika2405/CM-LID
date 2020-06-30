import argparse
import torch
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scikitplot.plotters as skplt
from dataloader import preprocess, hash
import pandas as pd
import numpy as np
from os import listdir
from model import LID
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_pred, y_target):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_target, y_pred)
    
    # precision tp / (tp + fp)
    precision = precision_score(y_target, y_pred)
    
    # recall: tp / (tp + fn)
    recall = recall_score(y_target, y_pred)
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_target, y_pred)
    
    return accuracy, precision, recall, f1

def compute_confusion_matrix(y_pred, y_target):
    matplotlib.rcParams.update({'text.color' : "black",
                     'axes.labelcolor' : "black"})
    #Confusion matrix
    cm = confusion_matrix(y_target, y_pred)
    np.set_printoptions(suppress=True)
    df_cm = pd.DataFrame(cm, index = [i for i in ['Hindi', 'English']],
                  columns = [i for i in ['Hindi', 'English']])

    sns.heatmap(df_cm, annot=True, fmt='g')
    # skplt.plot_confusion_matrix(y_target, y_pred,figsize=(8,8))
    # plt.xticks([0,1],['Hindi','English'])
    # plt.yticks([0,1],['Hindi','English'])
    plt.savefig('confusion_matrix.png')
    print('Saved confusion matrix!')

def predict(data, target):

    embedding_dim = 256
    hidden_dim = 256
    input_dim  = 11240

    lid = LID(embedding_dim, hidden_dim, input_dim)
    pred = lid(data)
    
    y_pred = torch.max(pred, dim=1).indices
    print('pred', y_pred)

    accuracy, precision, recall, f1 = compute_metrics(y_pred, target)
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)

    compute_confusion_matrix(y_pred, target)

    return y_pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', help='Path of saved model')
    parser.add_argument('--ngrams_path', help='Path for ngrams')
    parser.add_argument('--vocab_path', help='Vocabulary path')
    parser.add_argument('--labels', help='Labels path')

    args = parser.parse_args()

    ngrams = [] 
    vocab_idx = [] 
    for i, j in zip(sorted(listdir(args.ngrams_path)), sorted(listdir(args.vocab_path))):
        ngrams.append(pd.read_pickle(args.ngrams_path + i))
        vocab_idx.append(pd.read_pickle(args.vocab_path + j))

    en_hi = pd.read_pickle(args.labels)
    labels = en_hi.lang
    word = en_hi.word

    dim = [240, 1000, 5000, 5000]

    dataset_size = len(labels)
    shuffle_dataset = True
    random_seed= 42

    indices = list(range(dataset_size))
    split = 10000
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    
    data = [preprocess(ngrams, index, vocab_idx, dim) for index in val_indices]
    labels = np.where(labels.iloc[val_indices]=='hin', 0, 1)

    pred = predict(torch.tensor(data).float(), torch.tensor(labels).long())

    results = pd.DataFrame(list(zip(word.iloc[val_indices], labels, pred.numpy())), columns = ['Word', 'Actual', 'Predicted'])
    results.to_csv('results.csv', index=False)







