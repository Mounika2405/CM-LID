import argparse
import re
import collections
from collections import Counter
from sklearn.model_selection import train_test_split
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Path of original data')
parser.add_argument('--train_path', help='Path for saving train split')
parser.add_argument('--test_path', help='Path for saving test split')

class Corpus():
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load(self):
        if self.data_path[len(self.data_path)-3: ] == 'txt':
            with open(self.data_path) as f:
                data = f.readlines()

            sentence = ''
            sentences = []
            lang = []
            labels = []

            for line in data:
                if line != '\n':
                    sentence = sentence + ' ' + line[ :line.find('\t')]
                    lang.append((line[line.find('\t')+1:[m.start() for m in re.finditer(r"\t",line)][1]]))
                else:
                    sentences.append(sentence.lstrip())
                    sentence = ''
                    labels.append(lang)
                    lang = []

            data = {'text': sentences, 'labels': labels}

            return data
        elif self.data_path[len(self.data_path) - 3: ] == 'pki': #tokenwise transliterated data
            tokens_data = pd.read_pickle(self.data_path)
            #Ignoring samples greater than length 12
            tokens_data = tokens_data[tokens_data['word'].apply(lambda x: len(x) <= 17)]
            data = {'text': list(tokens_data.word), 'labels': list(tokens_data.lang)}
            return data

    def split_data(self, X, y):
        X_train, X_val, y_train, y_val = train_val_split(X, y, val_size=0.33, random_state=42, shuffle=True)
        self.train_dataset = {'text': X_train, 'labels': y_train}
        self.val_dataset = {'text': X_val, 'labels': y_val}
        return self.train_dataset, self.val_dataset


class Ngrams():
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.indices = {}

    def fit(self):
        words = self.data['text']
        labels = self.data['labels']
        ngrams = []
        i = len(self.indices)-1

        for word in tqdm(words):
            word = "^" + word + "$" #Appending additional boundary symbol at both ends of the token
            for x in zip(*[word[i:] for i in range(self.n)]):
                if self.indices.get(x) == None:
                    i+=1
                    self.indices.update({x:i})

    def transform(self, data):
        words = data['text']
        vecs = []
        
        for word in tqdm(words):
            vec = []
            word = "^" + word + "$" #Appending additional boundary symbol at both ends of the token
            for x in zip(*[word[i:] for i in range(self.n)]):
                v = [0] * len(self.indices)
                if self.indices.get(x) != None:                                                                                                                              
                    v[self.indices[x]] = 1
                vec.append(v)
            count = dict(Counter(tuple(ng) for ng in vec))
            length = len(vec)

            #Averages the embeddings according to the fractions of each n-gram string in the input token.
            final_vec = []
            for i in range(len(count)):
                final_vec.append(list(count.values())[i]/length)
            vecs.append(final_vec)
        return vecs

def vectorize_labels(tokens, to_ix):
    idx = []
    for w in tqdm(tokens):
        idx.extend([to_ix[w]])
    return idx

def pad(vecs):
    length_vecs = [len(w) for w in vecs]
    padded_vecs = []
    for w in range(len(vecs)):
        pad = [5] * (57 - length_vecs[w]) #(max(length_vecs))=57
        l = vecs[w]
        l.extend(pad)
        padded_vecs.append(l)
    return padded_vecs

class LexiconFeatures():
    def __init__(self, data):
        words = data['text']
        lang = data['labels']

        hi = dict(Counter(list(np.array(words)[np.where(np.array(lang) == 'hin')[0]])))
        en = dict(Counter(list(np.array(words)[np.where(np.array(lang) == 'eng')[0]])))
        vocab_dict = dict.fromkeys(set(words),0)
        
        dicts = [vocab_dict, hi]
        # sum the values with same keys 
        counter = collections.Counter() 
        for d in dicts:  
            counter.update(d) 
        self.hi = dict(counter)    

        dicts = [vocab_dict, en]
        # sum the values with same keys 
        counter = collections.Counter() 
        for d in dicts:  
            counter.update(d) 
        self.en = dict(counter) 

    def compute_lang_dist(self, data):
        words = data['text']
        labels = data['labels']

        lang_dist = []        
        for word in tqdm(words):
            try:
                hi_dist = self.hi[word]
            except KeyError:
                hi_dist = 0.0
            try:
                en_dist = self.en[word]
            except KeyError:
                en_dist = 0.0
   
            num_labels = len(np.unique(labels))
            lang_dist.append([hi_dist/num_labels, en_dist/num_labels])
        return lang_dist

    def compute_alang_dist(self, data):
        words = data['text']
        labels = data['labels']

        active_lang = []
        for word in tqdm(words):
            alang = [0] * len(np.unique(labels))
            try:
                hi_dist = self.hi[word]
            except KeyError:
                hi_dist = 0.0
            try:
                en_dist = self.en[word]
            except KeyError:
                en_dist = 0.0  

            if hi_dist > 0:
                alang[0] = 1
            if en_dist > 0:
                alang[1] = 1
            active_lang.append(alang)
        return active_lang

def main():
    args = parser.parse_args()
    corpus = Corpus(args.data_path)
    data = corpus.load()

    train_data, test_data = corpus.split_data(data['text'], data['labels'])

    if any(isinstance(el, list) for el in train_data['text']):
        word_list = [word for line in train_data['text'] for word in line.split()]
        num_tokens = len(word_list) 
        vocab = set(word_list)   
    else:
        num_tokens = len(train_data['text'])
        vocab = set(train_data['text'])
    
    vocab_size = len(vocab)
    print('Total number of tokens:', num_tokens)
    print('Vocabulary size:', vocab_size)

    # sizes = []
    # for word in vocab:
    #     sizes.append(len(word))
    
    # print('Longest word', max(sizes))

    for n in range(3):
        trvecs = []
        tevecs = []

        ngrams = Ngrams(train_data, n+1)
        print('Fitting ngrams model for n =', n+1)
        ngrams.fit()

        print('Computing', n+1, '- gram features for train set...')
        ngram_train = ngrams.transform(train_data)

        if n == 0:
            ngram_train_vecs = [[] for i in range(len(ngram_train))]
        for i, j in zip(ngram_train_vecs, ngram_train):
            trvecs.append(i + j)
        ngram_train_vecs = trvecs

        print('Computing', n+1, '- ngram features for test set...')
        ngram_test = ngrams.transform(test_data)

        if n == 0:
            ngram_test_vecs = [[] for i in range(len(ngram_test))]
        for i, j in zip(ngram_test_vecs, ngram_test):
            tevecs.append(i + j)
        ngram_test_vecs = tevecs
    
    tag_to_ix = {"hin": 0, "eng": 1}
    print('Vectorizing train labels...')
    train_label_vecs = vectorize_labels(train_data['labels'], tag_to_ix)
    print('Vectorizing test labels...')
    test_label_vecs = vectorize_labels(test_data['labels'], tag_to_ix)

    lex = LexiconFeatures(train_data)

    print('Computing language distribution for train set...')
    langdist_train_vecs = lex.compute_lang_dist(train_data) 
    print('Computing language distribution for test set...')
    langdist_test_vecs = lex.compute_lang_dist(test_data)
 
    print('Computing active lang distribution for train set...')
    activelang_train_vecs = lex.compute_alang_dist(train_data)

    print('Computing active lang distribution for test set')
    activelang_test_vecs = lex.compute_alang_dist(test_data)
    
    #Concatenating features
    train_vecs = []
    for ldist_features, alang_features, ngram_features in zip(langdist_train_vecs, activelang_train_vecs, ngram_train_vecs):
        train_vecs.append([ldist_features +  alang_features + ngram_features])

    test_vecs = []
    for ldist_features, alang_features, ngram_features in zip(langdist_test_vecs, activelang_test_vecs, ngram_test_vecs):
        test_vecs.append(ldist_features + alang_features + ngram_features)

    print('train_vecs', len(train_vecs))
    print('test_vecs', len(test_vecs))

    train_vecs = pad(train_vecs)
    test_vecs = pad(test_vecs)


    train_split = {'text': train_vecs, 'labels': train_label_vecs}
    test_split = {'text': test_vecs, 'labels': test_label_vecs}

    with open(args.train_path, 'w') as f:
      f.write(json.dumps(train_split))
    print('Saved train split')

    with open(args.test_path, 'w') as f:
      f.write(json.dumps(test_split))

    print('Saved test split')

if __name__ == '__main__':
    main()



