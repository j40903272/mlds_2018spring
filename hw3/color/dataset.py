import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
import json
import itertools as it
import random
from collections import Counter
from PIL import Image
from utils import hair2idx, eyes2idx

class ColorDataset(Dataset):

    def __init__(self):
        print('Loading dataset')
        with open('../fn2tags', 'rb') as f:
            fn2tags = pickle.load(f)
            
        imgs, tags = [], []
        for i in fn2tags:
            img = np.array(Image.open(os.path.join('../', i)))
            tagtext = fn2tags[i]
            
            for h in hair2idx:
                if h in tagtext:
                    break

            for e in eyes2idx:
                if e in tagtext:
                    break
                    
            imgs.append(img)
            
            h, e = hair2idx[h], eyes2idx[e]
            tag = [h, e] if e != 0 else [h, 1]
            tags.append(tag)
        
        
        imgs, tags = np.array(imgs)/255, np.array(tags)
        data = [(torch.from_numpy(img).float(), LongTensor(tag)) for img, tag in zip(imgs, tags)]
        self.data = data
        print (len(data), 'pairs of [img,tags]')

    def collate_fn(self, batch):
        imgs, tags = zip(*batch)
        imgs = torch.stack(imgs, 0)
        tags = torch.stack(tags, 0)
        return imgs, tags

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
