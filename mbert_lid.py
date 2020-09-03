#Import libraries
import random
import numpy as np
import ast
import pandas as pd
import argparse
import os
import json
import itertools
from tqdm import tqdm, trange
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

#Read data
def read_data(args):
    if args.data_path[-3:] == 'txt':
        with open(args.data_path) as f:
            data = f.readlines()
        data = ast.literal_eval(data[0])
        final_data = []
        for t, l in zip(data['text'], data['labels']):
            d = {'word': [t], 'labels':l}
            final_data.append(d)
        return final_data
    elif args.data_path[-3:] == 'pkl':
        words = []
        data = pd.read_pickle(args.data_path)
        final_data = []
        for i, row in data.iterrows():
            d = {'word': [row['word']], 'labels':[row['lang']]}
            final_data.append(d)
            words.append(row['word'])
        args.words = words
        return final_data

#Prepare data for BERT
def prepare_data(examples, tokenizer, mode):
    #Split the sentence to tokens, add the special [CLS] & [SEP] tokens and map tokens to their ids in mbert vocab
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    print('Tokenizing and assigning IDs to text...', flush=True)
    input_ids = []
    label_ids = []
    # for sentence in tqdm(sentences):
    #     encoded_sent = tokenizer.encode(sentence, add_special_tokens = True)
    #     # Add the encoded sentence to the list.
    #     input_ids.append(encoded_sent)

    MAX_LEN = 64
    label_map = {'en': 0, 'eng':0, 'Eng': 0, 'hi': 1, 'hin': 1, 'Hin':1, 'rest': 2, 'PAD': 3}
    pad_value = CrossEntropyLoss().ignore_index   # features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        sentence = []
        labels = []
        if(len(example['labels']) > 1):
            for word, label in zip(example['word'], example['labels']):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    sentence.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    labels.extend([label_map[label]]
                                  + [pad_value] * (len(word_tokens) - 1))
                    assert len(sentence) == len(labels)
        elif len(example['labels']) == 1:
            word_tokens = tokenizer.tokenize(example['word'][0])
            if len(word_tokens) > 0:
                sentence.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                labels.extend([label_map[example['labels'][0]]]
                              + [pad_value] * (len(word_tokens) - 1))
                assert len(sentence) == len(labels)
        sentence_tokens = sentence[:MAX_LEN - 2]
        label_id = labels[:MAX_LEN - 2]
        sentence_tokens = [tokenizer.cls_token] + \
            sentence_tokens + [tokenizer.sep_token]
        label_id = [pad_value] + label_id + [pad_value]
        input_id = tokenizer.convert_tokens_to_ids(sentence_tokens)
        assert len(input_id) == len(label_id)
        input_ids.append(input_id)
        label_ids.append(label_id)

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN, flush=True)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=0, truncating="post", padding="post")
    print('Done.', flush=True)

    print('\nPadding/truncating all labels to %d values...' % MAX_LEN, flush=True)
    tags = pad_sequences(label_ids, maxlen=MAX_LEN, dtype="long", 
                          value=pad_value, truncating="post", padding="post")
    print('Done.', flush=True)

    #Attention masks(to specify which tokens are actual words(1) versus which are padding(0))
    print('Creating attention masks...', flush=True)
    attention_masks = []
    for sent in tqdm(input_ids):
        att_mask = [int(token_id > 0) for token_id in sent] 
        attention_masks.append(att_mask)

    return input_ids, tags, attention_masks

#Create splits for train and val and prepare a dataloader
def create_splits(input_ids, tags, attention_masks):
    X_train, X_val, y_train, y_val = train_test_split(input_ids, tags,
                                                                random_state=2020, test_size=0.1)
    masks_train, masks_val, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2020, test_size=0.1)
    return X_train, X_val, y_train, y_val, masks_train, masks_val

def create_dataloader(X, masks, y, bs):
    X = torch.tensor(X)
    y = torch.tensor(y)
    masks = torch.tensor(masks)
    data = TensorDataset(X, masks, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs, shuffle=False)
    return dataloader

def plot_results(args, train_results, val_results):
    plt.figure(figsize=(8,8))
    plt.plot(list(range(args.num_epochs)), train_results, label='Train Loss')
    plt.plot(list(range(args.num_epochs)), val_results, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve for Fine-tuned mBERT Model')
    plt.legend()
    plt.savefig(args.output_dir + args.data_path[5:-4] + '_loss_curve.png')

def save_results(args, metrics, words, actual, pred):
    with open(args.output_dir + '_metrics.txt', 'a') as f:
        f.write('***** Metrics for ' + args.data_path[5:-4] + ' *****')
        f.write('\n')
        f.write(json.dumps(metrics))
        f.write('\n\n')
    words = list(itertools.chain.from_iterable(words))
    results = pd.DataFrame(list(zip(words, actual, pred)), columns = ['Word', 'Actual', 'Predicted'])
    results.to_csv(args.output_dir + '_' + args.data_path[5:-4] + '_results.csv', index=False)
    matplotlib.rcParams.update({'text.color' : "black",
                     'axes.labelcolor' : "black"})
    #Confusion matrix
    cm = confusion_matrix(actual, pred)
    np.set_printoptions(suppress=True)
    df_cm = pd.DataFrame(cm, index = [i for i in ['English', 'Hindi', 'Rest']],
                  columns = [i for i in ['English', 'Hindi', 'Rest']])
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(args.output_dir + '_' + args.data_path[5:-4] + '_confusion_matrix.png')
    print('Saved results')

#Create model architecture
def train(args, train_dataloader, val_dataloader, model, tokenizer):
    # Prepare optimizer
    print('Length of train_dataloader', len(train_dataloader), flush=True)
    total_steps = len(train_dataloader) * args.num_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Training
    print('Training...', flush=True)
    global_step = 0
    train_loss = 0.0
    train_losses, eval_losses = [], []
    model.zero_grad()
    train_iterator = trange(int(args.num_epochs), desc="Epoch")
    set_seed(args)
    best_f1_score = 0

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      'attention_mask': batch[1],
                      "labels": batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            loss_t = loss.item()
            train_loss += (loss_t - train_loss) / (step + 1)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        # Checking for validation accuracy and stopping after drop in accuracy for 3 epochs
        results = evaluate(args, val_dataloader, model, tokenizer, 'validation')
        if results.get('f1') > best_f1_score and args.save_steps > 0:
            best_f1_score = results.get('f1')
            model_to_save = model.module if hasattr(model, "module") else model
            print('Saving checkpoints...', flush = True)
            model_to_save.save_pretrained(args.output_dir + '.pt')
            tokenizer.save_pretrained(args.output_dir + '.pt')
            torch.save(args, os.path.join(
                args.output_dir + '.pt', "training_args.bin"))
            print('Done.', flush=True)
        train_losses.append(train_loss)
        eval_losses.append(results['loss'])
    print('train_losses', train_losses, flush = True)
    print('eval_loss', eval_losses, flush=True)
    plot_results(args, train_losses, eval_losses)
    
#Write code for Evaluation metrics and plots
def evaluate(args, dataloader, model, tokenizer, prefix=""):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    input_ids = []
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[2]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()
            # eval_loss += (loss_t - eval_loss) / (nb_eval_steps + 1)
        nb_eval_steps += 1
        preds.extend([t for t in logits.detach().cpu()])
        out_label_ids.extend([t for t in inputs["labels"].detach().cpu()])
        input_ids.extend([t for t in inputs["input_ids"].detach().cpu()])

    # eval_losses.append(eval_loss)
    eval_loss = eval_loss / nb_eval_steps
    preds = [np.argmax(t, axis=1) for t in preds]
    label_map = {0: 'en', 1: 'hi', 2: 'rest', 3: 'PAD'}
    out_label_list = [[] for _ in range(len(out_label_ids))]
    preds_list = [[] for _ in range(len(out_label_ids))]
    tokens_list = [[] for _ in range(len(out_label_ids))]
    pad_token_label_id = CrossEntropyLoss().ignore_index


    for i in range(len(out_label_ids)):
        for j in range(out_label_ids[i].shape[0]):
            if out_label_ids[i][j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j].item()])
                preds_list[i].append(label_map[preds[i][j].item()])

    words = []
    for t in input_ids:
        words.append(tokenizer.convert_ids_to_tokens(t))
    for i, t in enumerate(words):
        word = ''
        for j in t:   
            if j.startswith("##"):
                word = word + j[2:]      
            elif j != '[PAD]' and j!= '[SEP]' and j != '[CLS]':
                word = word + j
        tokens_list[i].append(word)

    out_label_list = list(itertools.chain.from_iterable(out_label_list))
    preds_list = list(itertools.chain.from_iterable(preds_list))
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list, average='weighted'),
        "recall": recall_score(out_label_list, preds_list, average='weighted'),
        "f1": f1_score(out_label_list, preds_list, average='weighted'),
        "accuracy": accuracy_score(out_label_list, preds_list)
    }

    print("***** Eval results %s *****", prefix, flush=True)
    for key in sorted(results.keys()):
      print("  %s = %s", key, str(results[key]), flush=True)

    if args.mode == 'test':
        save_results(args, results, tokens_list, out_label_list, preds_list)
    return results

#Initialisation and calling functions
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', help='Data path')
    parser.add_argument('--mode', help='train/test')
    parser.add_argument('--bs', help='Batch size', default = 32, type=int)
    parser.add_argument('--num_epochs', help='Number of epochs', default = 1000, type=int)
    parser.add_argument("--lr", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--output_dir", help='Path for saving output files')
    parser.add_argument("--model_name", help='Model path', default='bert-base-multilingual-cased')
    parser.add_argument("--save_steps", type=int, default=1, help='set to -1 to not save model')
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    data= read_data(args)
    num_labels = 3

    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    input_ids, tags, attention_masks = prepare_data(data, tokenizer, args.mode)

    if args.mode == 'train':
        X_train, X_val, y_train, y_val, masks_train, masks_val= create_splits(input_ids, tags, attention_masks)
        train_dataloader = create_dataloader(X_train, masks_train, y_train, args.bs)
        val_dataloader = create_dataloader(X_val, masks_val, y_val, args.bs)
    elif args.mode == 'test':
        test_dataloader = create_dataloader(input_ids, attention_masks, tags, args.bs)
    model = BertForTokenClassification.from_pretrained(args.model_name,
                                                            num_labels = num_labels,
                                                            output_attentions = False,
                                                            output_hidden_states = False)
        
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for mBERT  model!", flush=True)
        watch = torch.nn.DataParallel(model)
    else:
        print("Using single GPU...", flush=True)

    model = model.to(device)
    if args.mode == 'train':
        train(args, train_dataloader, val_dataloader, model, tokenizer)
    elif args.mode == 'test':
        evaluate(args, test_dataloader, model, tokenizer)

if __name__ == '__main__':
    main()
