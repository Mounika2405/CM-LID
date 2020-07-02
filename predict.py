import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    plt.savefig('confusion_matrix-test-dictdata.png')
    print('Saved confusion matrix!')

def predict(data, target):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_dim = 256
    hidden_dim = 256
    input_dim  = 10897

    lid = LID(embedding_dim, hidden_dim, input_dim)

    ckpt = torch.load(args.model_path)
    lid.load_state_dict(ckpt['model_state_dict'])
    lid.to(device)

    lid.eval()

    data = data.to(device)
    pred = lid(data)
    
    y_pred = torch.max(pred, dim=1).indices


    accuracy, precision, recall, f1 = compute_metrics(y_pred.cpu(), target)
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)

    compute_confusion_matrix(y_pred.cpu(), target)

    return y_pred.cpu()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', help='Path of saved model')
    parser.add_argument('--data_path', help='Path for ngrams')
    parser.add_argument('--vocab_path', help='Vocabulary path')

    args = parser.parse_args()

    vocab_idx = [] 
    for i in sorted(listdir(args.vocab_path)):
        vocab_idx.append(pd.read_pickle(args.vocab_path + i))

    data = pd.read_pickle(args.data_path)
    ngrams = [data['unigram'], data['bigram'], data['trigram'], data['4gram']]
    labels = data.lang
    word = data.word

    dim = [43, 854, 5000, 5000]
    
    features = [preprocess(ngrams, index, vocab_idx, dim) for index in range(len(data))]
    labels = np.where(labels == 'hin', 0, 1)

    pred = predict(torch.tensor(features).float(), torch.tensor(labels).long())

    results = pd.DataFrame(list(zip(word, labels, pred.numpy())), columns = ['Word', 'Actual', 'Predicted'])
    results.to_csv('results-test-dictdata.csv', index=False)







