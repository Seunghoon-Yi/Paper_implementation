import model
import dataloader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy, time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(CurrEpoch, Model, iterator, optimizer, metric, clip):
    Model.train()
    epoch_loss = 0

    for i, (features, translations) in enumerate(iterator):
        src, trg = features.to(device), translations.to(device)# 마찬가지로 같은 gpu 위에 올려줘야

        optimizer.zero_grad()  # Initialize gradient
        output = Model(src, trg[:, :-1])
        # output: [BS, trg_len - 1, output_dim]
        # trg:    [BS, trg_len]
        output_dim = output.shape[-1]

        # Re-allocate memory :
        # output: [BS, trg_len - 1, output_dim]
        # trg:    [BS, trg_len-1], exclude <SOS>
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = metric(output, trg)
        loss.backward()
        # And gradient clipping :
        torch.nn.utils.clip_grad_norm_(Model.parameters(), clip)
        # Update params :
        optimizer.step()
        # total loss in epoch
        epoch_loss += loss.item()
        # Print intra-epoch loss
        #print(f'{CurrEpoch} / Step {i + 1} : Loss = {loss.item()}')

    return epoch_loss / len(iterator)


def eval(Model, iterator, metric):  # No gradient updatd, no optimizer and clipping
    Model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (features, translations) in enumerate(iterator):
            src, trg = features.to(device), translations.to(device)

            output = Model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = metric(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Define dataloader, datasets
path = 'C:/Users/PC/2021-MLVU/SLT_project/'
train_loader, dataset, pad_idx, train_vocab_size = dataloader.get_loader(path,
                            annotation_file=path + "PHOENIX-2014-T.train.corpus.csv")
test_loader, dataset, pad_idx, test_vocab_size = dataloader.get_loader(path,
                            annotation_file=path + "PHOENIX-2014-T.train.corpus.csv")
val_loader, dataset, pad_idx, val_vocab_size = dataloader.get_loader(path,
                            annotation_file=path + "PHOENIX-2014-T.train.corpus.csv")
Output_dim = train_vocab_size  # Since we're only training the model on the training dataset!

#################################### Current Transformer ###################################
#            Input(BS, max_len, 1024)        |-------------------->|
#               |             |              |           Decoder Layers x n
#               X      +     PE              |                     |
#                      |                     |      Output(BS, seq_len, vocab_size)
#             Encoder Layers x n             |
#       encodings(BS, max_len, 1024)         |
#                      |-------------------->|
############################################################################################
Transformer = model.Transformer(
    Output_dim, Output_dim, pad_idx, pad_idx, device = device).to(device)

# Count parameters and initialize weights
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(Transformer):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

Transformer.apply(initialize_weights)

# Hyperparameters
lr_init = 0.0005
BATCH_SIZE = 16
optimizer = torch.optim.Adam(Transformer.parameters(), lr = lr_init)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training func
def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_epoch = 50
Clip = 2
best_val_loss = float("inf")

Trainloss, Valloss = [], []

for epoch in range(N_epoch):
    start = time.time()
    print(device)
    train_loss = train(epoch+1, Transformer, train_loader, optimizer, criterion, Clip)
    val_loss   = eval(Transformer, val_loader, criterion)

    Trainloss.append(train_loss) ; Valloss.append(val_loss)

    end = time.time()
    epoch_m, epoch_s = epoch_time(start, end)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(Transformer.state_dict(), 'transformer_de_to_en.pt')

    print(f'Epoch {epoch + 1:02} | Time : {epoch_m}m, {epoch_s}s')
    print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}')
    print(f'\t Val Loss : {val_loss:.3f} | Val PPL : {math.exp(val_loss):.3f}')
