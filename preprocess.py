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

class Preprocess():
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_dataset(self):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
        self.train_dataset = {'text': X_train[:40000], 'labels': y_train[:40000]}
        self.test_dataset = {'text': X_test[40000:60000], 'labels': y_test[40000:60000]}
    #         validation_split = .2
    #         shuffle_dataset = True
    #         random_seed= 42

    #         # Creating data indices for training and validation splits:
    #         dataset_size = len(dataset)
    #         indices = list(range(dataset_size))
    #         split = int(np.floor(validation_split * dataset_size))
    #         if shuffle_dataset :
    #             np.random.seed(random_seed)
    #             np.random.shuffle(indices)
    #         train_indices, val_indices = indices[split:], indices[:split]

    #         # Creating PT data samplers and loaders:
    #         train_sampler = SubsetRandomSampler(train_indices)
    #         valid_sampler = SubsetRandomSampler(val_indices)
    #         return train_sampler, valid_sampler
        return self.train_dataset, self.test_dataset

    


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

        # for sentence in sentences:
        #     words = sentence.split()
        #     grams = []
        
        for word in tqdm(words):
            word = "^" + word + "$" #Appending additional boundary symbol at both ends of the token
            for x in zip(*[word[i:] for i in range(self.n)]):
                if self.indices.get(x) == None:
                    i+=1
                    self.indices.update({x:i})

    def transform(self, data):
#         self.fit()
        words = data['text']
        ngrams = []
#         print(self.indices)
        
        # vectors = []
        # for sentence in sentences:
        #     words = sentence.split()
        grams = []
        vecs = []
        
        for word in tqdm(words):
            vec = []
            word = "^" + word + "$" #Appending additional boundary symbol at both ends of the token
            for x in zip(*[word[i:] for i in range(self.n)]):
#                     print(x)
                v = [0] * len(self.indices)
                if self.indices.get(x) != None:
                                                                                                                                
                    v[self.indices[x]] = 1
                vec.append(v)
            count = dict(Counter(tuple(ng) for ng in vec))
            length = len(vec)

            final_vec = []
            for i in range(len(count)):

                final_vec.append(list(count.values())[i]/length)
            
            vecs.append(final_vec)
        
        length_vecs = [len(w) for w in vecs]
        
        final_vecs = []
        for w in range(len(vecs)):
            # pad = [0] * ((max(length_vecs)) - length_vecs[w]) 
            pad = [10] * (17 - length_vecs[w]) #15 is the max length of words #check
            l = vecs[w]
            l.extend(pad)

            final_vecs.append(l)
        # vectors.append(final_vecs)
        return final_vecs

def vectorize_labels(tokens, to_ix):
    # idxs = []
    # for seq in seqs:
    idx = []
    
    for w in tqdm(tokens):
        idx.extend([to_ix[w]])
    # idxs.append(idx)

    return idx

class LexiconFeatures():
    def __init__(self, data):
        words = data['text']
        lang = data['labels']

        # words = []
        # for sent in sentences:
        #     words.extend(sent.split())
        # lang = sum(labels, [])

        hi = dict(Counter(list(np.array(words)[np.where(np.array(lang) == 'hin')[0]])))
        en = dict(Counter(list(np.array(words)[np.where(np.array(lang) == 'eng')[0]])))
        # rest = dict(Counter(list(np.array(words)[np.where(np.array(lang) == 'rest')[0]])))

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

        # dicts = [vocab_dict, rest]
          
        # # sum the values with same keys 
        # counter = collections.Counter() 
        # for d in dicts:  
        #     counter.update(d) 
              
        # self.rest = dict(counter)


    def compute_lang_dist(self, data):
        words = data['text']
        labels = data['labels']
        # lang_dists = []

        # for sentence in sentences:
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
            # try:
            #     rest_dist = self.rest[word]
            # except KeyError:
            #     rest_dist = 0.0    
            num_labels = len(np.unique(labels))
            lang_dist.append([hi_dist/num_labels, en_dist/num_labels])
            

        # lang_dists.append(lang_dist)

        return lang_dist

    def compute_alang_dist(self, data):
        words = data['text']
        labels = data['labels']

        active_lang = []
        # active_langs = []
        # for sentence in sentences:
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
            # try:
            #     rest_dist = self.rest[word]
            # except KeyError:
            #     rest_dist = 0.0    

            if hi_dist > 0:
                alang[0] = 1
            if en_dist > 0:
                alang[1] = 1
            # if rest_dist > 0:
            #     alang[2] = 1
            active_lang.append(alang)
        # active_langs.append(active_lang)
        return active_lang

def main():
    args = parser.parse_args()
    preprocess = Preprocess(args.data_path)
    data = preprocess.load_dataset()

    train_data, test_data = preprocess.split_data(data['text'], data['labels'])
    

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
    # for n in range(3):
    ngrams = Ngrams(train_data, 3)
    print('Fitting ngrams model...')
    ngrams.fit()

    print('Computing ngram features for train set...')
    ngram_train_vecs = ngrams.transform(train_data)
    print('Computing ngram features for test set...')
    ngram_test_vecs = ngrams.transform(test_data)

    tag_to_ix = {"hin": 0, "eng": 1}
    print('Vectorizing train labels...')
    train_label_vecs = vectorize_labels(train_data['labels'], tag_to_ix)
    print('Vectorizing test labels...')
    test_label_vecs = vectorize_labels(test_data['labels'], tag_to_ix)
    # print('Label vectorization done')

    lex = LexiconFeatures(train_data)

    print('Computing language distribution for train set...')
    langdist_train_vecs = lex.compute_lang_dist(train_data) 
    print('Computing language distribution for train set...')
    langdist_test_vecs = lex.compute_lang_dist(test_data)
    # print('Language distributions computed')
 
    print('Computing active lang distribution for train set...')
    activelang_train_vecs = lex.compute_alang_dist(train_data)

    print('Computing active lang distribution for test set')
    activelang_test_vecs = lex.compute_alang_dist(test_data)
    # print('Active language distributions computed')
    
    
    #Concatenating features
    train_vecs = []
    for ldist_features, alang_features, ngram_features in zip(langdist_train_vecs, activelang_train_vecs, ngram_train_vecs):
        train_vecs.append([ldist_features +  alang_features + ngram_features])

    test_vecs = []
    for ldist_features, alang_features, ngram_features in zip(langdist_test_vecs, activelang_test_vecs, ngram_test_vecs):
        test_vecs.append(ldist_features + alang_features + ngram_features)

    print('train_vecs', len(train_vecs))
    print('test_vecs', len(test_vecs))

    #Eliminating empty lists
    # train_vecs = [x for x in train_vecs if x]
    # test_vecs = [x for x in test_vecs if x]
    # train_label_vecs = [x for x in train_label_vecs if x]
    # test_label_vecs = [x for x in test_label_vecs if x]

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



