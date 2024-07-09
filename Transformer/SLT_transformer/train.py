from model import Transformer_
from dataloader import Vocab_tokenizer, get_loader
from bleu import calc_BLEU
from sklearn.utils import shuffle
import pandas as pd
import os

import numpy as np
import torch
import torch.nn as nn
from Scheduler import CosineAnnealingWarmUpRestarts
import torch.nn.functional as F
import math, copy, time
import itertools




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(CurrEpoch, Model, iterator, optimizer, scheduler, metric, metric_translation, clip,
          lam_translation, lam_gloss):
    Model.train()
    epoch_loss = 0

    for i, (features, glosses, translations) in enumerate(iterator):
        src, orth, trg = \
            features.to(device), glosses.to(device), translations.to(device)
        #print(glosses.shape, translations.shape)

        optimizer.zero_grad()  # Initialize gradient

        predict_translation, predict_gloss = Model(src, trg[:, :-1])
        #print(predict_translation.shape, trg.shape)
        translation_dim = predict_translation.shape[-1]
        gloss_dim = predict_gloss.shape[-1]

        predict_translation = predict_translation.contiguous().view(-1, translation_dim)
        predict_gloss = predict_gloss.contiguous().view(-1, gloss_dim)
        # GTs
        # print(orth)
        orth = orth.contiguous().view(-1)
        orth = orth.type(torch.LongTensor).to(device)
        trg = trg[:, 1:].contiguous().view(-1)
        trg = trg.type(torch.LongTensor).to(device)

        loss_translation = metric(predict_translation, trg)
        loss_gloss = metric(predict_gloss, orth)

        loss = (lam_translation * loss_translation + lam_gloss * loss_gloss) / (lam_gloss + lam_translation)
        loss.backward()

        # And gradient clipping :
        torch.nn.utils.clip_grad_norm_(Model.parameters(), clip)
        # Update params :
        optimizer.step()
        scheduler.step()
        # total loss in epoch
        epoch_loss += loss.item()

        #print("+"*50)

    return epoch_loss / len(iterator)


def eval(CurrEpoch, Model, iterator, metric, data_tokenizer, lam_translation, lam_gloss):  # No gradient updatd, no optimizer and clipping
    Model.eval()
    epoch_loss = 0

    with torch.no_grad():
        test_sentence = []
        GT_sentence = []
        for i, (features, glosses, translations) in enumerate(iterator):
            src, orth, trg = \
                features.to(device), glosses.to(device), translations.to(device)

            predict_translation, predict_gloss = Model(src, trg[:, :-1])

            # Generate text file
            for tokens in predict_translation:
                # Get argmax of tokens, bring it back to CPU.
                tokens = torch.argmax(tokens, dim=1).to(dtype=torch.long, device=torch.device("cpu"))
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

            translation_dim = predict_translation.shape[-1]
            gloss_dim = predict_gloss.shape[-1]

            # Re-allocate memory :
            # output: [BS, trg_len - 1, output_dim]
            # trg:    [BS, trg_len-1], exclude <SOS>
            # Predictions
            predict_translation = predict_translation.contiguous().view(-1, translation_dim)
            predict_gloss = predict_gloss.contiguous().view(-1, gloss_dim)
            # GTs
            orth = orth.contiguous().view(-1)
            orth = orth.type(torch.LongTensor).to(device)
            trg = trg[:, 1:].contiguous().view(-1)
            trg = trg.type(torch.LongTensor).to(device)

            loss_translation = metric(predict_translation, trg)
            loss_gloss = metric(predict_gloss, orth)

            loss = (lam_translation * loss_translation + lam_gloss * loss_gloss) / (lam_gloss + lam_translation)
            epoch_loss += loss.item()

        #print(test_sentence, '\n', GT_sentence)
        BLEU4 = calc_BLEU(test_sentence, GT_sentence)

    return epoch_loss / len(iterator), BLEU4

'''

            # Predictions
            len_trans_pred = torch.Tensor([len(trans) for trans in predict_translation]).type(torch.LongTensor)
            log_prob_trans = predict_translation.contiguous().log_softmax(2)
            len_orth_pred = torch.Tensor([len(orth_) for orth_ in predict_gloss]).type(torch.LongTensor)
            log_prob_orth = predict_gloss.contiguous().log_softmax(2)

            # GTs
            orth_opt = orth.contiguous().type(torch.LongTensor).to(device)
            trg_opt = trg[:, 1:].contiguous().type(torch.LongTensor).to(device)
            len_orth_ipt = torch.Tensor([(sum(t > 0 for t in gloss)) for gloss in orth_opt]).type(torch.LongTensor)
            len_trans_ipt = torch.Tensor([(sum(t > 0 for t in trans))-1 for trans in trg_opt]).type(torch.LongTensor)
            # Loss
            loss_translation = metric(log_prob_trans.permute(1, 0, 2), trg_opt, len_trans_pred, len_trans_ipt)
            loss_gloss = metric(log_prob_orth.permute(1, 0, 2), orth_opt, len_orth_pred, len_orth_ipt)'''


def translate(Model, iterator, metric, data_tokenizer, max_len = 55):
    Model.eval()
    with torch.no_grad():
        test_sentence = []
        GT_sentence = []
        for i, (features, glosses, translations) in enumerate(iterator):
            src, orth, trg = \
                features.to(device), glosses.to(device), translations.to(device)

            src_mask = Model.make_source_mask(src)
            enc_feature, predict_gloss = Model.Encoder(src, src_mask)

            trg_index = [[data_tokenizer.stoi["<SOS>"]] for i in range(src.size(0))]
            #print(trg_index)
            for j in range(max_len):
                #print(torch.LongTensor(trg_index).shape)
                trg_tensor = torch.LongTensor(trg_index).to(device)
                trg_mask = Model.make_target_mask(trg_tensor)
                output   = Model.Decoder(trg_tensor, enc_feature, src_mask, trg_mask)
                output   = nn.Softmax(dim=-1)(output)

                pred_token = torch.argmax(output, dim=-1)[:,-1]
                #print(torch.argmax(output, dim=-1))

                for target_list, pred in zip(trg_index, pred_token.tolist()):
                    target_list.append(pred)

            # Generate text file
            for tokens in trg_index:
                # Get argmax of tokens, bring it back to CPU.
                #print(tokens)
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

            #print(torch.Tensor(trg_index).shape)

        #print(test_sentence, '\n', GT_sentence)
        BLEU4 = calc_BLEU(test_sentence, GT_sentence)

    return BLEU4



# Count parameters and initialize weights
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    ################################## Prepare datasets and dataloader ##################################
    data_path = 'C:/Users/PC/2021-MLVU/SLT_project/'
    train_data = pd.read_csv(data_path + "PHOENIX-2014-T.train.corpus.csv", delimiter='|')
    val_data   = pd.read_csv(data_path + "PHOENIX-2014-T.dev.corpus.csv", delimiter='|')
    test_data  = pd.read_csv(data_path + "PHOENIX-2014-T.test.corpus.csv", delimiter='|')

    Traindata = pd.concat([train_data, val_data])
    max_len   = 55

    # Define the tokenizer. data : translation, orth : gloss
    data_tokenizer = Vocab_tokenizer(freq_th=1, max_len = max_len)
    orth_tokenizer = Vocab_tokenizer(freq_th=1, max_len = max_len)

    data_tokenizer.build_vocab(Traindata.translation)
    orth_tokenizer.build_vocab(Traindata.orth)
    #print(orth_tokenizer.stoi)

    targets = data_tokenizer.numericalize(Traindata.translation)
    glosses = orth_tokenizer.numericalize(Traindata.orth)
    labels  = Traindata.name.to_numpy()

    print("Translation : ", targets.shape, len(data_tokenizer),
          "\n", "Glosses ; ", glosses.shape, len(orth_tokenizer))    # (7615, 300) 2948

    #labels, targets, glosses = shuffle(labels, targets, glosses, random_state = 42)
    # Train and validation (feature labels, translations)
    train_labels, train_glosses, train_translations = labels[:7140], glosses[:7140], targets[:7140]
    val_labels,   val_glosses,   val_translations   = labels[7140:], glosses[7140:], targets[7140:]
    # test ''
    test_labels       = test_data.name.to_numpy()
    test_glosses      = orth_tokenizer.numericalize(test_data.orth)
    test_translations = data_tokenizer.numericalize(test_data.translation)

    lr_init = 4.e-6
    decay_factor = 0.996
    BATCH_SIZE = 32
    N_epoch = 175
    Clip = 1
    L_translate = [1, 1, 1, 1, 2]
    L_orth = [10, 5, 2, 1, 1]
    dropout_list = [0.2]

    #lr = [1.2e-4, 1.5e-4, 2.e-4]
    #n_layer = [3, 4]

    for drop_rate, (l_tr, l_orth) in itertools.product(dropout_list, zip(L_translate, L_orth)):

        # Define loader and datasets
        train_loader, train_dataset, pad_idx = get_loader(
            data_path, train_labels, train_glosses, train_translations, n_workers=2, BS=BATCH_SIZE)
        val_loader, val_dataset, pad_idx = get_loader(
            data_path, val_labels, val_glosses, val_translations, n_workers=2, BS=BATCH_SIZE)
        test_loader, test_dataset, pad_idx = get_loader(
            data_path, test_labels, test_glosses, test_translations, n_workers=2, BS=BATCH_SIZE)
        N_tokens = len(data_tokenizer)  # Since we're only training the model on the training dataset!
        N_glosses = len(orth_tokenizer)

        #l_tr, l_orth                     = 1, 5
        lr_max, n_layers                 = 1.e-4, 3
        dropout                          = drop_rate
        Trainloss, Valloss, BLUE4_scores = [], [], []
        best_BLEU4_score                 = -float("inf")


        Transformer = Transformer_(N_glosses, N_tokens, pad_idx, pad_idx,
                                   n_layers=n_layers, dropout=0.2, device = device).cuda()
        print(f'The model has {count_parameters(Transformer):,} trainable parameters')
        print('-'*10, f'n_layers = {n_layers}, dropout = {drop_rate}, traslation : gloss = {l_tr} : {l_orth}', '-'*10)

        # Weight Initialization
        Transformer.apply(initialize_weights)
        # Define optimizer and lr scheduler
        optimizer = torch.optim.Adam(Transformer.parameters(), lr = lr_init)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=7115//BATCH_SIZE*26, T_mult=2, eta_max=lr_max,  T_up=7115//BATCH_SIZE*5, gamma=0.8)

        # LOSS functions! #
        criterion       = nn.CrossEntropyLoss().cuda() # nn.CTCLoss(blank=0, reduction='sum').cuda()
        criterion_gloss = nn.CTCLoss(blank=0, reduction='mean').cuda()

        ############## Training body ##############
        for epoch in range(N_epoch):
            start = time.time()
            print(device)
            train_loss = train(epoch+1, Transformer, train_loader, optimizer, scheduler,
                               criterion, criterion_gloss, Clip, l_tr, l_orth)
            val_loss, BLUE4_score = eval(epoch+1, Transformer, val_loader, criterion, data_tokenizer, l_tr, l_orth)
            test_BLUE4_score = translate(Transformer, test_loader, criterion, data_tokenizer, max_len=max_len)

            Trainloss.append(train_loss) ;BLUE4_scores.append(BLUE4_score); Valloss.append(val_loss)

            end = time.time()
            epoch_m, epoch_s = epoch_time(start, end)

            if test_BLUE4_score > best_BLEU4_score and (epoch > 5):
                best_BLEU4_score = test_BLUE4_score
                torch.save(Transformer.state_dict(), f'B{BATCH_SIZE}_drop_{dropout}_tr_{l_tr}_gl_{l_orth}.pt')

            print('lr = ', optimizer.param_groups[0]['lr'])

            print(f'Epoch {epoch + 1:02} | Time : {epoch_m}m, {epoch_s}s')
            print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {train_loss:.3f}')
            print(f'\t Val Loss : {val_loss:.3f} | Val PPL : {math.exp(val_loss):.3f}')
            print(f'\t Val BLEU4 : {BLUE4_score:.3f}')
            print(f'\t test BLEU4 : {test_BLUE4_score:.3f}')

        # Print and store losses #
        #print("translation : gloss = ", f'{l_tr} : {l_orth}')
        print(min(Trainloss), min(Valloss), max(BLUE4_scores))

        print("Done, sleep for 500 seconds \n")
        time.sleep(500)

if __name__ == "__main__":
    main()