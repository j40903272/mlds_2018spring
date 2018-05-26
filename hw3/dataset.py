import os
import numpy as np
import cv2
from utils import illum

import torch
from torch.utils.data import Dataset
from torch import LongTensor, FloatTensor


class Data(Dataset):
    
    def __init__(self):
        print('Loading dataset')
        
        imgs = []
        for dir_path in ['../faces', '../extra_data/images', '../faces2']:
            cnt = 0
            for fn in os.listdir(dir_path):
                img = cv2.imread(os.path.join(dir_path, fn))
                if img.shape != (96, 96, 3):
                    continue
                imgs.append(img)
                cnt += 1
            print (dir_path, cnt)
            #break
        
        imgs = np.array(imgs) / 255
        
        data = []
        for img in imgs:
            I = illum(img)
            t = I[8:24, 16:-16]
            l = t >= np.percentile(t, 25)
            u = t <= np.percentile(t, 75)
            tag = float(np.mean(t[np.logical_and(l, u)]))
            data.append((I, tag))
        
        data = [(torch.from_numpy(img).float().unsqueeze(0), FloatTensor([tag])) for img, tag in data]
        self.data = data
        print (len(data), 'images')

    def collate_fn(self, batch):
        imgs, tags = zip(*batch)
        tags = torch.stack(tags, 0).squeeze(1)
        imgs = torch.stack(imgs, 0)
        return imgs, tags

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)