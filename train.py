
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
parser.add_argument('--test_data_path', help='Path of test data(vectorized) file')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--embed_dim', default=100, type=int, help='Dimensions of embedding vectors')
parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs')
parser.add_argument('--bs', default=256, type=int, help='Batch size')
parser.add_argument('--vocab_size', default=37566, type=int, help='Size of vocabulary')
parser.add_argument('-rm', '--resume', default=None, \
                        help='Path of the saved model to resume training')
parser.add_argument('-cf', '--ckpt_freq', type=int, default=1, required=False, \
                        help='Frequency of saving the model')

def compute_accuracy(y_pred, y_target):



    y_pred_indices = torch.max(y_pred, dim=1).indices
    n_correct = torch.eq(y_pred_indices, y_target).sum(dim=1)


    n_acc = torch.mean(torch.div(n_correct.float(), y_pred_indices.shape[1]).float())

    return n_acc * 100

def plot_results(args, train_results, val_results, type):
    plt.figure(figsize=(8,8))
    plt.plot(list(range(args.num_epochs)), train_results, label='Train ' + type)
    plt.plot(list(range(args.num_epochs)), val_results, label='Validation ' + type)

    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.legend()
    plt.savefig(type + '_curve.png')

def main():

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    embedding_dim = args.embed_dim #check
    hidden_dim = 256
    vocab_size = args.vocab_size #check

    lid = LID(embedding_dim, hidden_dim, vocab_size, device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for Watch model!")
        watch = nn.DataParallel(lid)
    else:
        print("Using single GPU")
    lid = lid.to(device)

    train_dataset = CMDataset(args.train_data_path)
    test_dataset = CMDataset(args.test_data_path)

    #The network is trained per-token with cross-entropy loss


    criterion = nn.CrossEntropyLoss(ignore_index=10)
    momentum=0.9 #incomplete #check
    optimizer = torch.optim.ASGD(lid.parameters(), lr=args.lr)
    decayRate = 0.96 #incomplete #check
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    epoch_resume=0
    if(args.resume):
        checkpoint = torch.load(args.resume)
        
        epoch_resume = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

        print("Model resumed for training...")
        # print("Epoch: ", epoch_resume)
        # print("Loss: ", loss)

    model_params = sum(p.numel() for p in lid.parameters() if p.requires_grad)
    print("Model parameters: ", model_params)

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    

    for epoch in range(epoch_resume+1, args.num_epochs+1):

        print('Epoch', epoch)
        #Iterate over training dataset
        
        train_generator = generate_batches(train_dataset, batch_size=args.bs)

        train_loss = 0.0
        val_loss = 0.0
        train_accuracy = 0.0
        val_accuracy = 0.0

        lid.train()
        
        for batch_index, batch_dict in enumerate(train_generator):
            
            optimizer.zero_grad()

            inputs = batch_dict[0].data
            inputs = inputs.to(device)

            y_pred = lid(inputs)
            target = batch_dict[1].long()
            target = target.view(y_pred.shape[0], y_pred.shape[1])
            target = target.to(device)
            # print(y_pred)
            # print(target.shape)
            y_pred = y_pred.permute(0, 2, 1).float()
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
        val_generator = generate_batches(test_dataset, batch_size=args.bs)
        progress_bar = tqdm(val_generator)

        for batch_index, batch_dict in enumerate(progress_bar):
            
            #Compute the output
            inputs = batch_dict[0].data
            inputs = inputs.to(device)

            y_pred = lid(inputs)
            y_pred = y_pred.view(y_pred.shape[0], 1, y_pred.shape[1])
            y_pred = y_pred.permute(0, 2, 1).float()
            target = batch_dict[1].long()
            target = target.view(y_pred.shape[0], y_pred.shape[2])
            target = target.to(device)
            
            #Compute loss
            loss = criterion(y_pred, target)
            
            loss_t = loss.item()
            val_loss += (loss_t - val_loss) / (batch_index + 1) #check

            loss.backward()
            
            optimizer.step()

            accuracy_t = compute_accuracy(y_pred, target)
            val_accuracy += (accuracy_t - val_accuracy) / (batch_index + 1)

            progress_bar.set_postfix(train_loss=train_loss, val_loss=val_loss, train_acc=train_accuracy.item(),
                val_acc=val_accuracy.item())
            progress_bar.refresh()

        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        if epoch % args.ckpt_freq == 0:
            
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': lid.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': (train_loss / len(train_generator))
            }, 'model.pt')
        

    plot_results(args, train_losses, val_losses, type='Loss')
    plot_results(args, train_acc, val_acc, type='Accuracy')



if __name__ == '__main__':
    main()