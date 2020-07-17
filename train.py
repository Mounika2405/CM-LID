
import argparse
import torch
import torch.nn as nn

from model import LID
from preprocess import Preprocess
from dataloader import *

from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--train_data_path', help='Path of train data(vectorized) file')
parser.add_argument('--val_data_path', help='Path of test data(vectorized) file')
# parser.add_argument('--ngrams_path', help='Path for ngrams')
parser.add_argument('--vocab_path', help='Vocabulary path')
# parser.add_argument('--labels', help='Labels path')
parser.add_argument('--model_name', help='model name')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--embed_dim', default=256, type=int, help='Dimensions of embedding vectors')
parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs')
parser.add_argument('--bs', default=256, type=int, help='Batch size')
parser.add_argument('--input_dim', default=10899, type=int, help='Input Dimensions')
parser.add_argument('-rm', '--resume', default=None, \
                        help='Path of the saved model to resume training')
parser.add_argument('-cf', '--ckpt_freq', type=int, default=1, required=False, \
                        help='Frequency of saving the model')

def compute_accuracy(y_pred, y_target):
    
    y_pred_indices = torch.max(y_pred, dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def plot_results(num_epochs, args, train_results, val_results, type):
    plt.figure(figsize=(8,8))
    plt.plot(list(range(num_epochs)), train_results, label='Train ' + type)
    plt.plot(list(range(num_epochs)), val_results, label='Validation ' + type)

    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.legend()
    plt.savefig(type + '-' + str(args.lr) + '-' + str(args.bs) + '_lex_curve.png')

def main():

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_dim = args.embed_dim #check
    hidden_dim = 256
    input_dim = args.input_dim #check

    lid = LID(embedding_dim, hidden_dim, input_dim)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for LID  model!")
        watch = nn.DataParallel(lid)
    else:
        print("Using single GPU")
    lid = lid.to(device)

    train_dataset = CMDataset(args.train_data_path, args.vocab_path, is_train=True)
    val_dataset = CMDataset(args.val_data_path, args.vocab_path, is_train=False)
    # validation_split = .2
    # shuffle_dataset = True
    # random_seed= 42

    # # Creating data indices for training and validation splits:
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = 10000
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)


    train_generator = generate_batches(train_dataset, batch_size=args.bs)
    val_generator = generate_batches(val_dataset, batch_size=args.bs)

    #The network is trained per-token with cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    momentum=0.9 #incomplete #check
    optimizer = torch.optim.SGD(lid.parameters(), lr=args.lr)
    decayRate = 0.96 #incomplete #check
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    epoch_resume=0
    if(args.resume):
        checkpoint = torch.load(args.resume) 
        epoch_resume = checkpoint['epoch']
        lid.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print("Model resumed for training...")

    print(lid)
    model_params = sum(p.numel() for p in lid.parameters() if p.requires_grad)
    print("Model parameters: ", model_params)

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    min_val_loss = np.Inf
    n_epochs_stop = 313*5
    epochs_no_improve = 0
    early_stop = False


    for epoch in range(epoch_resume+1, args.num_epochs+1):

        print('Epoch', epoch)
        #Iterate over training dataset

        train_loss = 0.0
        val_loss = 0.0
        train_accuracy = 0.0
        val_accuracy = 0.0

        lid.train()
        
        for batch_index, batch_dict in enumerate(train_generator):
            
            optimizer.zero_grad()
            
            inputs = batch_dict[0].data.float()
            inputs = inputs.to(device)

            y_pred = lid(inputs)

            target = batch_dict[1].long()
            target = target.to(device)

            loss = criterion(y_pred, target)

            loss_t = loss.item()
            train_loss += (loss_t - train_loss) / (batch_index + 1) #check

            loss.backward()            
            optimizer.step()

            accuracy_t = compute_accuracy(y_pred, target)
            train_accuracy += (accuracy_t - train_accuracy) / (batch_index + 1)

        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        
        #Iterate over validation set

        lid.eval()
        
        progress_bar = tqdm(val_generator)

        for batch_index, batch_dict in enumerate(progress_bar):
            
            #Compute the output
            inputs = batch_dict[0].data.float()
            inputs = inputs.to(device)
            y_pred = lid(inputs)

            target = batch_dict[1].long()
            target = target.to(device)
            
            #Compute loss
            loss = criterion(y_pred, target)
            loss_t = loss.item()
            val_loss += (loss_t - val_loss) / (batch_index + 1) #check

            accuracy_t = compute_accuracy(y_pred, target)
            val_accuracy += (accuracy_t - val_accuracy) / (batch_index + 1)

            progress_bar.set_postfix(train_loss=train_loss, val_loss=val_loss, train_acc=train_accuracy,
                val_acc=val_accuracy)
            progress_bar.refresh()

            if val_loss < min_val_loss:

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': lid.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': (train_loss / len(train_generator)),
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy

                }, args.model_name)
                epochs_no_improve = 0
                min_val_loss = val_loss
      
            else:
                epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!' )
                early_stop = True
                break
            else:
                continue
            
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        if early_stop:
            print("Stopped")
            break
        
    plot_results(epoch, args, train_losses, val_losses, type='Loss')
    plot_results(epoch, args, train_acc, val_acc, type='Accuracy')

if __name__ == '__main__':
    main()
