import numpy as np
import torch
from torch.utils.data import Dataset


class Conversation(Dataset):

    def __init__(self, filename, vocab):

        self.wtoi = {}
        self.itow = []

        vocab = open(vocab, 'r', encoding='utf-8').read().splitlines()
        for i in vocab:
            idx, w = i.split()
            idx = int(idx)
            self.wtoi[w] = idx
            self.itow.append(w)

        corpus = open(filename, 'r', encoding='utf-8').read().splitlines()

        self.sentences = []
        self.len = []
        for s in corpus:
            seq = []
            for c in s:
                if c != ' ':
                    if c in self.wtoi:
                        seq.append(self.wtoi[c])
                    else:
                        seq.append(self.wtoi['<UNK>'])

            seq.append(self.wtoi['<EOS>'])
            self.len.append(len(seq))
            for j in range(74-len(seq)):
                seq.append(0)
            self.sentences.append(seq)

        self.offset = 0

    def __len__(self):
        return len(self.sentences) - 1

    def __getitem__(self, idx):
        return self.sentences[idx]


class DatasetIterator():
    def __init__(self, dataset):
        self.dataset = dataset
        self.offset = 0

    def next_batch(self, batch_size):
        if self.offset + batch_size > len(self.dataset):
            batch_size = len(self.dataset) - self.offset

        i = self.offset
        self.offset += batch_size
        self.offset = self.offset % len(self.dataset)
        return self.dataset[i:i+batch_size], self.dataset[i+1:i+1+batch_size], self.dataset.len[i:i+batch_size]
