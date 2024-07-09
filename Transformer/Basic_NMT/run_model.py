import torch
from torch import nn, Tensor
import torch.optim as optim
import spacy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import model
import time
import math

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k

# Spacy instructions : https://spacy.io/usage
spacy_de = spacy.load("de_core_news_sm")
spacy_en  = spacy.load("en_core_web_sm")
# Deustch tokenizer
def tok_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]
# English tokenizer
def tok_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

# Explicitely assert preprocessing process.
SRC = Field(tokenize=tok_de, init_token="<sos>", eos_token="<eos>",
            lower=True, batch_first=True)
TRG = Field(tokenize=tok_en, init_token="<sos>", eos_token="<eos>",
            lower=True, batch_first=True)

train, val, test = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))

# Build vocab with over 2 apperances
SRC.build_vocab(train, min_freq=2)
TRG.build_vocab(train, min_freq=2)
print(f"len(SRC): {len(SRC.vocab)}")
print(f"len(TRG): {len(TRG.vocab)}")

# Device and Batching, make iterator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train, val, test), batch_size = BATCH_SIZE, device = device)

#######################################################
#                    Training Section                 #
#######################################################
Input_dim = len(SRC.vocab)
Output_dim = len(TRG.vocab)
src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
# The model #
transformer_model = model.Transformer(
    Input_dim, Output_dim, src_pad_idx, trg_pad_idx, device = device).to(device)

# Weight initialization, model configuration #
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(transformer_model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

transformer_model.apply(initialize_weights)


# Training and Validation functions #
def train(CurrEpoch, Model, iterator, optimizer, metric, clip):
    Model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch.src, batch.trg

        optimizer.zero_grad()  # Initialize gradient
        output = Model(src.to(device), trg[:, :-1].to(device))
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
        print(f'{CurrEpoch} / Step {i + 1} : Loss = {loss.item()}')

    return epoch_loss / len(iterator)


def eval(Model, iterator, metric):  # No gradient updatd, no optimizer and clipping
    Model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch.src, batch.trg

            output = Model(src.to(device), trg[:, :-1].to(device))
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = metric(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

#######################################################
#                    Training Section                 #
#######################################################

lr_init = 0.0005
optimizer = torch.optim.Adam(transformer_model.parameters(), lr = lr_init)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_epoch = 3
Clip = 2
best_val_loss = float("inf")

for epoch in range(N_epoch):
    start = time.time()

    train_loss = train(epoch+1, transformer_model, train_iterator, optimizer, criterion, Clip)
    val_loss   = eval(transformer_model, val_iterator, criterion)

    end = time.time()
    epoch_m, epoch_s = epoch_time(start, end)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(transformer_model.state_dict(), 'transformer_de_to_en.pt')

    print(f'Epoch {epoch + 1:02} | Time : {epoch_m}m, {epoch_s}s')
    print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}')
    print(f'\t Val Loss : {val_loss:.3f} | Val PPL : {math.exp(val_loss):.3f}')
