import os
import numpy as np
import pandas as pd
import spacy                 # For tokenizer?
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import spacy.cli
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


#spacy.cli.download("en_core_web_md")

spacy_de = spacy.load('de_core_news_lg')

print("Build start")

path = 'C:/Users/PC/2021-MLVU/SLT_project/'

class Vocab:
    def __init__(self, freq_th):
        self.itos = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}
        self.stoi = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}
        self.freq_th = freq_th

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        '''
        :param text: input translation string
        :return:     separated, lower-cased string
        '''
        # ["i", "love", "you"] * BS
        return[tok.text.lower() for tok in spacy_de.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = {}
        idx = 4      # Since we pre-occupied three idx

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies :
                    frequencies[word] = 1
                else :
                    frequencies[word] += 1

                # Store the words over the frequency to self.itos and self.stoi
                if frequencies[word] == self.freq_th:
                    self.stoi[word] = idx
                    self.itos[idx]  = word
                    idx += 1

    # Integer tokenizer
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class SLT_dataset(Dataset):
    def __init__(self, root_dir, data_file, transform = None, freq_th = 1):
        self.root_dir = root_dir
        self.df       = pd.read_csv(data_file, delimiter='|')
        self.transform = transform

        # get images and caption columns
        self.features = self.df["name"].to_numpy()
        self.translations = self.df["translation"].to_numpy()

        # Initialize and build vocab
        self.vocab = Vocab(freq_th)
        self.vocab.build_vocab(self.translations.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        '''
        :param index: (int) index that we want to see
        :return:      (str, png?) : caption-figure pair of index
        '''
        feature_id = self.features[index]
        translation  = self.translations[index]
        # Load image
        span = 'span8_stride2/'
        # (seq_len, 1024)
        feature   = torch.stack(torch.load(os.path.join(self.root_dir, span, feature_id+'.pt')))

        # Convert text to tokens
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens+= self.vocab.numericalize(translation)
        tokens.append(self.vocab.stoi["<EOS>"])

        return feature, torch.tensor(tokens)

    def vocablen(self):
        return self.vocab.__len__()


class Padder:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        '''
        :param batch: batch images and texts
        :return:      padded batch images and padded texts
        '''

        # item : from __getitem__ : a single*BS (img, caption)
        features = [item[0] for item in batch]    # Additional dimension for batch
        features = pad_sequence(features, batch_first=True, padding_value=self.pad_idx)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return features.squeeze(2), targets

def get_loader(
        root_folder,
        annotation_file,
        BS = 32,
        n_workers = 0,
        shuffle = True,
        pin_memory = True,
        test_split = 0.2,
        random_seed = 42
):
    dataset = SLT_dataset(root_folder, annotation_file)
    vocab_len = dataset.vocablen()

    dataset_len = len(dataset)
    data_inedx  = list(range(dataset_len))
    #split_index = int(np.floor(dataset_len*test_split))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(data_inedx)
    train_indices = data_inedx

    # Initialize first token to "PAD"
    pad_idx = dataset.vocab.stoi["<PAD>"]
    # Define data sampler
    train_sampler = SubsetRandomSampler(train_indices)
    # Define dataloader. Be careful that sampler option is mutually exclusive with shuffle.
    train_loader = DataLoader(dataset=dataset, batch_size=BS,
                         num_workers=n_workers, sampler=train_sampler,
                         pin_memory=pin_memory,collate_fn=Padder(pad_idx=pad_idx))
    '''test_loader  = DataLoader(dataset=dataset, batch_size=BS,
                              num_workers=n_workers, sampler=test_sampler,
                              pin_memory=pin_memory,collate_fn=Padder(pad_idx=pad_idx))'''
    return train_loader, dataset, pad_idx, vocab_len



'''
def main():
    train_loader, dataset, pad_idx, vocab_size = get_loader(path,
                            annotation_file=path + "PHOENIX-2014-T.test.corpus.csv")

    for idx, (imgs, captions) in enumerate(train_loader):
        print(imgs.shape, imgs.type)
        print(captions.shape)
    print(pad_idx, vocab_size)

if __name__  == "__main__":
    main()'''