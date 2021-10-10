import itertools

from model import SLT_Transformer
from dataloader import Vocab_tokenizer, get_loader
from bleu import calc_BLEU
from sklearn.utils import shuffle
import pandas as pd
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Scheduler import CosineAnnealingWarmUpRestarts
import torch.nn.functional as F
import math, copy, time
import torchvision
from pytorch_model_summary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(CurrEpoch, Model, iterator, optimizer, scheduler, metric, metric_translation, clip,
          lam_translation, lam_gloss):
    Model.train()
    epoch_loss = 0

    for i, (frames, glosses, translations) in enumerate(iterator):
        src, orth, trg = \
            frames.to(device), glosses.to(device), translations.to(device)

        optimizer.zero_grad()  # Initialize gradient
        predict_translation, predict_gloss = Model(src, trg[:, :-1])
        translation_dim = predict_translation.shape[-1]
        gloss_dim       = predict_gloss.shape[-1]

        # Predictions
        predict_translation = predict_translation.contiguous().view(-1, translation_dim)
        predict_gloss       = predict_gloss.contiguous().view(-1, gloss_dim)

        # GTs
        orth = orth.contiguous().view(-1)
        orth = orth.type(torch.LongTensor).to(device)
        trg = trg[:, 1:].contiguous().view(-1)
        trg = trg.type(torch.LongTensor).to(device)

        loss_translation = metric(predict_translation, trg)
        #print(predict_gloss.shape, orth.shape)
        loss_gloss       = metric(predict_gloss, orth)

        if i == 100:
            print(loss_gloss)
        loss = 0
        if CurrEpoch < 25:  # First, train without learning padding indices
            loss = (loss_translation*lam_translation + loss_gloss*lam_gloss)/(lam_gloss + lam_translation)
            loss.backward()
        elif (25 <= CurrEpoch) and (CurrEpoch < 75):  # Then, learn the paddings
            loss = (loss_translation*lam_translation + loss_gloss*lam_gloss)/(lam_gloss + lam_translation)
            loss.backward()
        else:
            if CurrEpoch % 2 == 0:
                loss = (loss_translation*lam_translation + loss_gloss*lam_gloss)/(lam_gloss + lam_translation)
                loss.backward()
            else:
                loss = (loss_translation*lam_translation + loss_gloss*lam_gloss)/(lam_gloss + lam_translation)
                loss.backward()

        # And gradient clipping :
        torch.nn.utils.clip_grad_norm_(Model.parameters(), clip)
        # Update params :
        optimizer.step()
        scheduler.step()
        # total loss in epoch
        epoch_loss += loss.item()
        # Print intra-epoch loss
        if i%1000 == 0:
            print(f'{CurrEpoch} / Step {i} : Loss = {loss.item()}')

    return epoch_loss/len(iterator)



def eval(CurrEpoch, Model, iterator, metric, data_tokenizer, metric_translation):  # No gradient updatd, no optimizer and clipping
    Model.eval()
    epoch_loss = 0

    with torch.no_grad():
        total_len = len(iterator)
        test_sentence = []
        GT_sentence = []

        for i, (frames, glosses, translations) in enumerate(iterator):
            src, orth, trg = \
                frames.to(device), glosses.to(device), translations.to(device)

            predict_translation, predict_gloss = Model(src, trg[:, :-1])

            # Generate text file
            for tokens in predict_translation:
                # Get argmax of tokens, bring it back to CPU.
                tokens = torch.argmax(tokens, dim = 1).to(dtype = torch.long, device = torch.device("cpu"))
                tokens = tokens.numpy()
                # make string, append it to test_sentence
                itos = data_tokenizer.stringnize(tokens)
                pred_string = ' '.join(itos)
                test_sentence.append(pred_string)
            for tokens in trg:
                tokens = tokens.to(dtype=torch.long, device=torch.device("cpu"))
                tokens = tokens.numpy()
                # make string, append it to test_sentence
                itos = data_tokenizer.stringnize(tokens[1:])
                GT_string = ' '.join(itos)
                GT_sentence.append(GT_string)

            # Calculate loss value
            translation_dim = predict_translation.shape[-1]
            gloss_dim       = predict_gloss.shape[-1]

            # Predictions
            predict_translation = predict_translation.contiguous().view(-1, translation_dim)
            predict_gloss       = predict_gloss.contiguous().view(-1, gloss_dim)
            # GTs
            orth = orth.contiguous().view(-1)
            orth = orth.type(torch.LongTensor).to(device)
            trg = trg[:, 1:].contiguous().view(-1)
            trg = trg.type(torch.LongTensor).to(device)

            loss_translation = metric(predict_translation, trg)
            loss = loss_translation
            # The total loss
            epoch_loss += loss.item()

        BLEU4 = calc_BLEU(test_sentence, GT_sentence)

        return epoch_loss / len(iterator), BLEU4

###################### Hyperparameters, transformation and dataloaders ######################

# Count parameters and initialize weights
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Weight initialization
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# Training func
def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



############################### Finally, The training Section ###############################
def main():

    base_path = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/'

    ###################### Get the csv file that needed in training process ######################
    train_data = pd.read_csv(base_path + "PHOENIX-2014-T.train.corpus.csv", delimiter='|')
    val_data = pd.read_csv(base_path + "PHOENIX-2014-T.dev.corpus.csv", delimiter='|')
    test_data = pd.read_csv(base_path + "PHOENIX-2014-T.test.corpus.csv", delimiter='|')

    Traindata = pd.concat([train_data, val_data])  # Train+dev data
    max_len = 55  # Max translation length

    ############################## Define the tokenizer and build. ##############################
    data_tokenizer = Vocab_tokenizer(freq_th=1, max_len=max_len)
    orth_tokenizer = Vocab_tokenizer(freq_th=1, max_len=max_len+1)

    data_tokenizer.build_vocab(Traindata.translation)
    orth_tokenizer.build_vocab(Traindata.orth)

    # target : Translation, glosses : glosses, labels : filename
    targets = data_tokenizer.numericalize(Traindata.translation)
    glosses = orth_tokenizer.numericalize(Traindata.orth)
    labels = Traindata.name.to_numpy()

    print("Translation : ", targets.shape, len(data_tokenizer),
          "\n", "Glosses : ", glosses.shape, len(orth_tokenizer))  # (7615, 300) 2948

    ############################# Split them into Train and dev set #############################
    labels, targets, glosses = shuffle(labels, targets, glosses, random_state=42)

    train_labels, train_glosses, train_translations = labels[:7115], glosses[:7115], targets[:7115]
    val_labels, val_glosses, val_translations = labels[7115:], glosses[7115:], targets[7115:]
    test_labels       = test_data.name.to_numpy()
    test_glosses      = orth_tokenizer.numericalize(test_data.orth)
    test_translations = data_tokenizer.numericalize(test_data.translation)


    lr_init = 4.e-6
    lr_ = [5.e-5, 1.e-4, 2.e-4]
    n_layer = [2]
    decay_factor = 0.996
    BATCH_SIZE = 4
    N_epoch = 75
    Clip = 1
    # dropout : 0.15 / 0.25 / 0.3 with lr fixed
    # tr : gloss : 1:1 to 1:5

    transforms_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.25, saturation=0.25)]
    )

    train_loader, train_dataset, pad_idx = get_loader(base_path, train_labels, train_glosses,
                                                      train_translations, n_workers=4, BS=BATCH_SIZE, transform=transforms_train)
    val_loader, val_dataset, pad_idx = get_loader(base_path, val_labels, val_glosses,
                                                  val_translations, n_workers=4, BS=BATCH_SIZE, transform=None)
    test_loader, test_dataset, pad_idx = get_loader(base_path, test_labels, test_glosses,
                                                    test_translations, n_workers=4, BS=BATCH_SIZE, transform=None)

    N_tokens = len(data_tokenizer)  # Since we're only training the model on the training dataset!
    N_glosses = len(orth_tokenizer)
    encoder_type = 'C+R3D'

    for lr_max, n_layers in itertools.product(lr_, n_layer):
        ######################### Define the model and auxiliary functions #########################
        best_BLEU4_score                 = -float("inf")
        Trainloss, Valloss, BLUE4_scores = [], [], []
        l_tr, l_orth                     = 1, 5


        Transformer = SLT_Transformer(N_glosses, N_tokens, pad_idx, pad_idx, n_layers=n_layers, device=device).cuda()
        Transformer.apply(initialize_weights)
        print(f'The model has {count_parameters(Transformer):,} trainable parameters')

        #if os.path.exists('./' + f'B{BATCH_SIZE}_n3_d512_{encoder_type}.pt'):
        #    print("Loading state_dict...")
        #    Transformer.load_state_dict(torch.load(f'B{BATCH_SIZE}_n3_d512_{encoder_type}.pt'))

        ######################## Optimizer, Scheduler and Loss functions ########################
        optimizer = torch.optim.AdamW(Transformer.parameters(), lr=lr_init, betas = (0.9, 0.999), eps = 1e-06)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=7115 // BATCH_SIZE * 40, T_mult=2, eta_max=lr_max, T_up=7115 // BATCH_SIZE * 8, gamma=0.7)
        # LOSS functions! #
        criterion = nn.CrossEntropyLoss().cuda()  # nn.CTCLoss(blank=0).cuda() #
        criterion_translation = nn.CrossEntropyLoss(ignore_index=0).cuda()  # nn.CTCLoss( blank=0).cuda() #
        # Print the entire model#
        print(summary(Transformer, torch.randn(1, 224, 3, 228, 196).cuda(),
                      torch.randint(0, 54, (1, 54)).cuda(),show_input=True, show_hierarchical=True))
        print('-'*20, f"lr = {lr_max}, n_layer = {n_layers}, translation : gloss = {l_tr} : {l_orth}", '-'*20)


        for epoch in range(N_epoch):
            start = time.time()
            print(device)
            train_loss = train(epoch+1, Transformer, train_loader, optimizer,scheduler,
                               criterion, criterion_translation, Clip, l_tr, l_orth)
            val_loss, BLUE4_score = eval(epoch+1, Transformer, test_loader, criterion, data_tokenizer, criterion_translation)

            Trainloss.append(train_loss) ; Valloss.append(val_loss) ; BLUE4_scores.append(BLUE4_score)

            end = time.time()
            epoch_m, epoch_s = epoch_time(start, end)

            if BLUE4_score > best_BLEU4_score and (epoch > 5):
                best_BLEU4_score = BLUE4_score
                torch.save(Transformer.state_dict(), f'lr_{lr_max}_n{n_layers}_d512_{encoder_type}.pt')


            print('lr = ', optimizer.param_groups[0]['lr'])

            print(f'Epoch {epoch + 1:02} | Time : {epoch_m}m, {epoch_s}s')
            print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}')
            print(f'\t Val Loss : {val_loss:.3f} | Val PPL : {math.exp(val_loss):.3f}')
            print(f'\t Val BLEU4 : {BLUE4_score:.3f}')

            # Save loss figure #
            x_epoch = range(epoch+1)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax2 = ax.twinx()
            ax.plot(x_epoch, Trainloss, 'r-', label = "train loss")
            ax.plot(x_epoch, Valloss, 'b-', label = 'val loss')
            ax2.plot(x_epoch, BLUE4_scores, 'g-', label = 'BLEU4 score')

            ax.set_xlabel("epoch", fontsize = 13)
            ax.set_ylabel("loss", fontsize = 13)
            ax2.set_ylabel("Test BLEU4", fontsize = 13)

            ax.grid()
            ax.legend() ; ax2.legend()


            plt.savefig(f'lr_{lr_max}_n2_d512_{encoder_type}.png', dpi = 250)
            plt.close(fig)

        # Print and store losses #
        print("translation : gloss = ", f'{l_tr} : {l_orth}')
        print(min(Trainloss), min(Valloss), max(BLUE4_scores))

if __name__  == "__main__":
    main()